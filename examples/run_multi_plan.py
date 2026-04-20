import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import numpy as np
import networkx as nx
import itertools

from assets.load import load_assembly
from assets.save import clear_saved_sdfs
from .run_joint_plan import get_planner as get_path_planner

import time
import settings


class SequencePlanner:

    def __init__(self, asset_folder, assembly_dir):

        self.asset_folder = asset_folder
        self.assembly_dir = assembly_dir
        self.assembly_id = os.path.basename(assembly_dir)

        self.graph = nx.DiGraph()
        
        meshes, names = load_assembly(assembly_dir, return_names=True)

        part_ids = [name.replace('.obj', '') for name in names]
        for i in range(len(part_ids)):
            self.graph.add_node(part_ids[i])

        self.num_parts = len(part_ids)
        assert self.num_parts > 1
        self.max_seq_count = (1 + self.num_parts) * self.num_parts // 2 - 1

        self.success_status = ['Success', 'Start with goal']
        self.failure_status = ['Timeout', 'Failure', 'Rigid']
        self.valid_status = self.success_status + self.failure_status

    def draw_graph(self):
        import matplotlib.pyplot as plt
        nx.draw(self.graph, with_labels=True)
        plt.show()

    def plan_sequence(self, *args, **kwargs):
        raise NotImplementedError

    def plan_path(self, move_id, still_ids, planner_name, rotation, body_type, sdf_dx, collision_th, force_mag, frame_skip,
        max_time, max_depth, seed, render, record_path, save_dir, n_save_state, return_contact=False, verbose=False, two_angles=False):

        # Ensure move_id and still_ids are passed correctly if move_id is a tuple/list
        if isinstance(move_id, tuple) or isinstance(move_id, list):
            move_id_for_planner = list(move_id)
            move_id_str = '_'.join(move_id)
        else:
            move_id_for_planner = move_id
            move_id_str = str(move_id)

        path_planner = get_path_planner(planner_name)(
            self.asset_folder, self.assembly_dir,
            move_id_for_planner, still_ids,
            rotation, body_type, sdf_dx, collision_th, force_mag, frame_skip, save_sdf=True
        )
        status, t_plan, path = path_planner.plan(max_time, max_depth=max_depth, seed=seed, return_path=True, render=render, record_path=record_path, verbose=verbose, two_angles=two_angles)

        assert status in self.valid_status, f'unknown status {status}'
        if save_dir is not None:
            path_planner.save_path(path, save_dir, n_save_state)

        if return_contact:
            contact_parts = path_planner.get_contact_bodies(move_id_for_planner)
            return status, t_plan, contact_parts
        else:
            return status, t_plan


class RandomSequencePlanner(SequencePlanner):

    def plan_sequence(self, path_planner_name, rotation, body_type, sdf_dx, collision_th, force_mag, frame_skip,
        seq_max_time, path_max_time, seed, render, record_dir, save_dir, n_save_state, verbose=False, max_iterations=None):

        np.random.seed(seed)

        if render and record_dir is not None:
            os.makedirs(record_dir, exist_ok=True)

        seq_status = 'Failure'
        sequence = []
        seq_count = 0
        t_plan_all = 0

        while seq_count < self.max_seq_count:

            all_ids = list(self.graph.nodes)
            move_id = np.random.choice(all_ids)
            still_ids = all_ids.copy()
            still_ids.remove(move_id)

            record_path = None
            if record_dir is None:
                record_path = None
            elif render:
                record_path = os.path.join(record_dir, f'{self.assembly_id}', f'{seq_count}_{move_id}.gif')

            if save_dir is not None:
                curr_save_dir = os.path.join(save_dir, f'{self.assembly_id}', f'{seq_count}_{move_id}')
            else:
                curr_save_dir = None

            curr_seed = np.random.randint(self.max_seq_count)

            status, t_plan = self.plan_path(move_id, still_ids,
                path_planner_name, False, body_type, sdf_dx, collision_th, force_mag, frame_skip,
                path_max_time, curr_seed, render, record_path, curr_save_dir, n_save_state)
            assert status in self.valid_status
            
            if status in self.failure_status and rotation:
                status, t_plan_rot = self.plan_path(move_id, still_ids,
                    path_planner_name, True, body_type, sdf_dx, collision_th, force_mag, frame_skip,
                    path_max_time, curr_seed, render, record_path, curr_save_dir, n_save_state)
                t_plan += t_plan_rot

            t_plan_all += t_plan
            seq_count += 1

            if verbose:
                print(f'# trials: {seq_count} | Move id: {move_id} | Status: {status} | Current planning time: {t_plan} | Total planning time: {t_plan_all}')
            
            if status in self.success_status:
                self.graph.remove_node(move_id)
                sequence.append(move_id)

            if len(self.graph.nodes) == 1:
                seq_status = 'Success'
                break

            if t_plan_all > seq_max_time:
                seq_status = 'Timeout'
                break

        if verbose:
            print(f'Result: {seq_status} | Disassembled: {len(sequence)}/{self.num_parts - 1} | Total # trials: {seq_count} | Total planning time: {t_plan_all}')

        return seq_status, sequence, seq_count, t_plan_all


class QueueSequencePlanner(SequencePlanner):

    def plan_sequence(self, path_planner_name, rotation, body_type, sdf_dx, collision_th, force_mag, frame_skip,
        seq_max_time, path_max_time, seed, render, record_dir, save_dir, n_save_state, verbose=False, max_iterations=None):

        np.random.seed(seed)

        if render and record_dir is not None:
            os.makedirs(record_dir, exist_ok=True)

        seq_status = 'Failure'
        sequence = []
        seq_count = 0
        t_plan_all = 0

        active_queue = list(self.graph.nodes) # nodes going to try
        np.random.shuffle(active_queue)
        last_active_queue = active_queue.copy() # for termination
        inactive_queue = [] # nodes tried

        while seq_count < self.max_seq_count:

            all_ids = list(self.graph.nodes)
            move_id = active_queue.pop(0)
            still_ids = all_ids.copy()
            still_ids.remove(move_id)

            if record_dir is None:
                record_path = None
            elif render:
                record_path = os.path.join(record_dir, f'{self.assembly_id}', f'{seq_count}_{move_id}.gif')

            if save_dir is not None:
                curr_save_dir = os.path.join(save_dir, f'{self.assembly_id}', f'{seq_count}_{move_id}')
            else:
                curr_save_dir = None

            curr_seed = np.random.randint(self.max_seq_count)

            status, t_plan = self.plan_path(move_id, still_ids,
                path_planner_name, False, body_type, sdf_dx, collision_th, force_mag, frame_skip,
                path_max_time, curr_seed, render, record_path, curr_save_dir, n_save_state)
            assert status in self.valid_status
            
            if status in self.failure_status and rotation:
                status, t_plan_rot = self.plan_path(move_id, still_ids,
                    path_planner_name, True, body_type, sdf_dx, collision_th, force_mag, frame_skip,
                    path_max_time, curr_seed, render, record_path, curr_save_dir, n_save_state)
                t_plan += t_plan_rot

            t_plan_all += t_plan
            seq_count += 1

            if verbose:
                print(f'# trials: {seq_count} | Move id: {move_id} | Status: {status} | Current planning time: {t_plan} | Total planning time: {t_plan_all}')
            
            if status in self.success_status:
                self.graph.remove_node(move_id)
                sequence.append(move_id)
            else:
                inactive_queue.append([move_id, max_depth + 1])

            if verbose:
                print('Active queue:', active_queue)
                print('Inactive queue:', inactive_queue)

            if len(self.graph.nodes) == 1:
                seq_status = 'Success'
                break

            if len(active_queue) == 0:
                active_queue = inactive_queue.copy()
                inactive_queue = []
                if active_queue == last_active_queue:
                    break # failure
                last_active_queue = active_queue.copy()

            if t_plan_all > seq_max_time:
                seq_status = 'Timeout'
                break

        if verbose:
            print(f'Result: {seq_status} | Disassembled: {len(sequence)}/{self.num_parts - 1} | Total # trials: {seq_count} | Total planning time: {t_plan_all}')
            print(f'Sequence: {sequence}')

        return seq_status, sequence, seq_count, t_plan_all


class ProgressiveQueueSequencePlanner(SequencePlanner):

    def plan_path(self, move_id, still_ids, planner_name, rotation, body_type, sdf_dx, collision_th, force_mag, frame_skip,
        max_time, max_depth, seed, render, record_path, save_dir, n_save_state, return_contact=False, verbose=False, two_angles=False):
        
        # Ensure move_id and still_ids are passed correctly if move_id is a tuple/list
        if isinstance(move_id, tuple) or isinstance(move_id, list):
            move_id_for_planner = list(move_id)
            move_id_str = '_'.join(move_id)
        else:
            move_id_for_planner = move_id
            move_id_str = str(move_id)

        path_planner = get_path_planner(planner_name)(
            self.asset_folder, self.assembly_dir,
            move_id_for_planner, still_ids,
            rotation, body_type, sdf_dx, collision_th, force_mag, frame_skip, save_sdf=True, color_scheme=settings.color_scheme
        )
        status, t_plan, path = path_planner.plan(max_time, max_depth=max_depth, seed=seed, return_path=True, render=render, record_path=record_path, verbose=verbose, two_angles=two_angles)

        assert status in self.valid_status, f'unknown status {status}'
        if save_dir is not None:
            path_planner.save_path(path, save_dir, n_save_state)

        if return_contact:
            contact_parts = path_planner.get_contact_bodies(move_id_for_planner)
            return status, t_plan, contact_parts
        else:
            return status, t_plan

    def plan_sequence(self, path_planner_name, rotation, body_type, sdf_dx, collision_th, force_mag, frame_skip,
        seq_max_time, path_max_time, seed, render, record_dir, save_dir, n_save_state, max_iterations=None, verbose=False, two_angles=False, max_moving_parts=1):

        np.random.seed(seed)

        if render and record_dir is not None:
            os.makedirs(record_dir, exist_ok=True)

        seq_status = 'Rigid'
        sequence = []
        seq_count = 0
        t_plan_all = 0

        def generate_queue(nodes, depth, max_moving):
            queue = []
            max_k = min(max_moving, len(nodes) // 2)
            max_k = max(1, max_k) # ensure at least 1
            for k in range(1, max_k + 1):
                combos = list(itertools.combinations(nodes, k))
                np.random.shuffle(combos)
                for combo in combos:
                    queue.append((combo if k > 1 else combo[0], depth))
            return queue

        active_queue = generate_queue(list(self.graph.nodes), 1, max_moving_parts)
        inactive_queue = [] 

        if verbose:
            print(f'Initial active queue: {active_queue}')

        while True:

            all_ids = list(self.graph.nodes)
            move_id, max_depth = active_queue.pop(0)
            
            still_ids = all_ids.copy()
            if isinstance(move_id, tuple):
                for m_id in move_id:
                    still_ids.remove(m_id)
                move_id_str = '_'.join(move_id)
            else:
                still_ids.remove(move_id)
                move_id_str = str(move_id)

            record_path = None
            if record_dir is None:
                record_path = None
            elif render:
                record_path = os.path.join(record_dir, f'{self.assembly_id}', f'{seq_count}_{move_id_str}.gif')

            if save_dir is not None:
                curr_save_dir = os.path.join(save_dir, f'{self.assembly_id}', f'{seq_count}_{move_id_str}')
            else:
                curr_save_dir = None

            curr_seed = np.random.randint(self.max_seq_count)

            status, t_plan = self.plan_path(move_id, still_ids,
                path_planner_name, False, body_type, sdf_dx, collision_th, force_mag, frame_skip,
                path_max_time, max_depth, curr_seed, render, record_path, curr_save_dir, n_save_state, verbose=verbose, two_angles=two_angles)
            assert status in self.valid_status
            
            if status in self.failure_status and rotation:
                status, t_plan_rot = self.plan_path(move_id, still_ids,
                    path_planner_name, True, body_type, sdf_dx, collision_th, force_mag, frame_skip,
                    path_max_time, max_depth, curr_seed, render, record_path, curr_save_dir, n_save_state, verbose=verbose, two_angles=two_angles)
                t_plan += t_plan_rot

            t_plan_all += t_plan
            seq_count += 1

            if verbose:
                print(f'# trials: {seq_count} | Move id: {move_id_str} | Status: {status} | Current planning time: {t_plan} | Total planning time: {t_plan_all}')
            
            if status in self.success_status:
                if isinstance(move_id, tuple):
                    for m_id in move_id:
                        self.graph.remove_node(m_id)
                else:
                    self.graph.remove_node(move_id)
                sequence.append(move_id)
                
                # Regenerate queue for remaining parts starting at depth 1
                active_queue = generate_queue(list(self.graph.nodes), 1, max_moving_parts)
                inactive_queue = []
            else:
                inactive_queue.append([move_id, max_depth + 1])

            if verbose:
                print('Active queue:', active_queue)
                print('Inactive queue:', inactive_queue)

            if status != 'Rigid': # we want to check if every part is rigid
                seq_status = status

            if len(self.graph.nodes) <= 1:
                seq_status = 'Success'
                break

            if len(active_queue) == 0:
                if seq_status == 'Rigid':
                    print('A subset of the assembly is completely rigid. Terminating.')
                    seq_status = 'Failure'
                    break
                active_queue = inactive_queue.copy()
                inactive_queue = []

            if t_plan_all > seq_max_time:
                seq_status = 'Timeout'
                break

            if max_iterations is not None and all(x > max_iterations for _, x in active_queue + inactive_queue):
                if verbose:
                    print(f'Max iterations ({max_iterations}) exceeded for all remaining parts. Terminating.')
                seq_status = 'Timeout'
                break

        if verbose:
            print(f'Result: {seq_status} | Disassembled: {len(sequence)} | Total # trials: {seq_count} | Total planning time: {t_plan_all}')
            print(f'Sequence: {sequence}')

        return seq_status, sequence, seq_count, t_plan_all


def get_seq_planner(name):
    seq_planners = {
        'random': RandomSequencePlanner,
        'queue': QueueSequencePlanner,
        'prog-queue': ProgressiveQueueSequencePlanner,
    }
    assert name in seq_planners, f'invalid planner name {name}'
    return seq_planners[name]


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--id', type=str, required=True, help='assembly id (e.g. 00000)')
    parser.add_argument('--dir', type=str, default='multi_assembly', help='directory storing all assemblies')
    parser.add_argument('--rotation', default=False, action='store_true')
    parser.add_argument('--seq-planner', type=str, required=True, choices=['random', 'queue', 'prog-queue'])
    parser.add_argument('--path-planner', type=str, required=True, choices=['bfs', 'bk-rrt'])
    parser.add_argument('--body-type', type=str, default='sdf', choices=['bvh', 'sdf'], help='simulation type of body')
    parser.add_argument('--sdf-dx', type=float, default=0.05, help='grid resolution of SDF')
    parser.add_argument('--collision-th', type=float, default=1e-2)
    parser.add_argument('--force-mag', type=float, default=100, help='magnitude of force')
    parser.add_argument('--frame-skip', type=int, default=100, help='control frequency')
    parser.add_argument('--seq-max-time', type=float, default=3600, help='sequence planning timeout')
    parser.add_argument('--path-max-time', type=float, default=120, help='path planning timeout')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--max-moving-parts', type=int, default=1, help='max number of parts to move simultaneously')
    parser.add_argument('--render', default=False, action='store_true', help='if render the result')
    parser.add_argument('--record-dir', type=str, default=None, help='directory to store rendering results')
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--n-save-state', type=int, default=100)
    parser.add_argument('--max-iterations', type=int, default=None)
    parser.add_argument('--verbose', action='store_true', help='whether to print detailed logs during planning')
    parser.add_argument('--use-previous-sdf', action='store_true', help='whether to use previously saved SDFs for faster planning')
    args = parser.parse_args()

    asset_folder = os.path.join(project_base_dir, './assets')
    assembly_dir = os.path.join(asset_folder, args.dir, args.id)

    if args.rotation: args.seq_max_time *= 2
        
    if not args.use_previous_sdf:
        clear_saved_sdfs(assembly_dir)

    if args.verbose:
        print(f'Running sequence planner')
    seq_planner = get_seq_planner(args.seq_planner)(asset_folder, assembly_dir)
    seq_status, sequence, seq_count, t_plan = seq_planner.plan_sequence(args.path_planner, 
        args.rotation, args.body_type, args.sdf_dx, args.collision_th, args.force_mag, args.frame_skip,
        args.seq_max_time, args.path_max_time, args.seed, args.render, args.record_dir, args.save_dir, args.n_save_state, verbose=args.verbose, max_iterations=args.max_iterations, max_moving_parts=args.max_moving_parts)
    #clear_saved_sdfs(assembly_dir)
    print(f'Final result for assembly {args.id}: {seq_status} | Sequence: {sequence}')