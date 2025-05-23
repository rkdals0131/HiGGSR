import numpy as np
import time
import multiprocessing
from scipy.spatial import KDTree
from numba import jit

from .feature_extraction import apply_transform_to_keypoints_numba

def select_diverse_candidates(candidates_info, num_to_select, separation_factor, cell_size_x, cell_size_y, map_x_range, map_y_range):
    if not candidates_info:
        return []

    valid_candidates = [c for c in candidates_info if c[0] is not None and c[0] > -float('inf')]
    if not valid_candidates:
        return []
        
    try:
        sorted_candidates = sorted(valid_candidates, key=lambda x: (x[0], x[1], x[2]), reverse=True)
    except TypeError as e:
        print(f"Error during candidate sorting in select_diverse_candidates: {e}. Candidates: {valid_candidates[:3]}")
        return [] 
    except IndexError as e:
        print(f"Error (IndexError) during candidate sorting: {e}. Candidates: {valid_candidates[:3]}")
        return []

    selected_candidates_final = []
    min_separation_dist_x = separation_factor * cell_size_x
    min_separation_dist_y = separation_factor * cell_size_y

    for cand_data in sorted_candidates:
        if len(selected_candidates_final) >= num_to_select:
            break
        
        cand_cx, cand_cy = cand_data[4], cand_data[5]
        
        is_far_enough = True
        if not np.isfinite(cand_cx) or not np.isfinite(cand_cy):
            continue 

        for sel_cand_data in selected_candidates_final:
            sel_cx, sel_cy = sel_cand_data[4], sel_cand_data[5]
            if not np.isfinite(sel_cx) or not np.isfinite(sel_cy):
                 continue 

            if abs(cand_cx - sel_cx) < min_separation_dist_x and \
               abs(cand_cy - sel_cy) < min_separation_dist_y:
                is_far_enough = False
                break
        
        if is_far_enough:
            selected_candidates_final.append(cand_data)
            
    return selected_candidates_final


def count_correspondences_kdtree(transformed_scan_keypoints_np, global_map_kdtree, distance_threshold):
    if transformed_scan_keypoints_np.shape[0] == 0 or global_map_kdtree is None: return 0
    distances, _ = global_map_kdtree.query(transformed_scan_keypoints_np, k=1, distance_upper_bound=distance_threshold)
    return np.sum(np.isfinite(distances))


def process_super_grid_cell(args):
    super_cx, super_cy, \
    live_scan_keypoints_np, global_map_kdtree, \
    actual_local_search_tx_half_width, actual_local_search_ty_half_width, \
    tx_search_steps, ty_search_steps, \
    theta_candidates_rad, theta_candidates_deg, \
    actual_correspondence_dist_thresh = args

    tx_candidates = np.linspace(super_cx - actual_local_search_tx_half_width, super_cx + actual_local_search_tx_half_width, tx_search_steps)
    ty_candidates = np.linspace(super_cy - actual_local_search_ty_half_width, super_cy + actual_local_search_ty_half_width, ty_search_steps)
    
    cell_best_score = -1
    cell_best_tx, cell_best_ty, cell_best_theta_deg = 0, 0, 0
    iterations_in_cell = 0

    for tx_candidate in tx_candidates:
        for ty_candidate in ty_candidates:
            for k_theta, theta_rad_candidate in enumerate(theta_candidates_rad):
                iterations_in_cell += 1
                transformed_scan_kps = apply_transform_to_keypoints_numba(
                    live_scan_keypoints_np, tx_candidate, ty_candidate, theta_rad_candidate
                )
                if transformed_scan_kps.shape[0] == 0: continue
                current_score = count_correspondences_kdtree(transformed_scan_kps, global_map_kdtree, actual_correspondence_dist_thresh)
                if current_score > cell_best_score:
                    cell_best_score = current_score
                    cell_best_tx = tx_candidate
                    cell_best_ty = ty_candidate
                    cell_best_theta_deg = theta_candidates_deg[k_theta]
    
    return cell_best_score, cell_best_tx, cell_best_ty, cell_best_theta_deg, iterations_in_cell, (super_cx, super_cy)


def search_single_level(
    level_idx, 
    global_map_keypoints_np, live_scan_keypoints_np,
    search_area_x_edges, search_area_y_edges, 
    grid_division, 
    theta_candidates_rad, theta_candidates_deg, 
    correspondence_dist_thresh, 
    tx_ty_search_steps_per_cell, 
    base_grid_cell_size, 
    global_map_kdtree, 
    num_processes
):
    print(f"  Executing search_single_level for Level {level_idx+1}...")
    map_x_min, map_x_max = search_area_x_edges[0], search_area_x_edges[-1]
    map_y_min, map_y_max = search_area_y_edges[0], search_area_y_edges[-1]

    if map_x_min >= map_x_max or map_y_min >= map_y_max:
        print(f"    Warning: Invalid search area for Level {level_idx+1}. Min/Max: X({map_x_min:.2f},{map_x_max:.2f}), Y({map_y_min:.2f},{map_y_max:.2f}). Skipping search.")
        return [], (np.array([map_x_min, map_x_max]), np.array([map_y_min, map_y_max])), 0

    num_super_x, num_super_y = grid_division 
    
    super_x_edges = np.linspace(map_x_min, map_x_max, num_super_x + 1)
    super_y_edges = np.linspace(map_y_min, map_y_max, num_super_y + 1)
    
    if num_super_x == 0: super_x_edges = np.array([map_x_min, map_x_max])
    if num_super_y == 0: super_y_edges = np.array([map_y_min, map_y_max])
    
    if num_super_x > 0:
        super_cx_centers = (super_x_edges[:-1] + super_x_edges[1:]) / 2
    else: 
        super_cx_centers = np.array([(map_x_min + map_x_max) / 2]) 
        super_x_edges = np.array([map_x_min, map_x_max]) 
        num_super_x = 1 

    if num_super_y > 0:
        super_cy_centers = (super_y_edges[:-1] + super_y_edges[1:]) / 2
    else:
        super_cy_centers = np.array([(map_y_min + map_y_max) / 2])
        super_y_edges = np.array([map_y_min, map_y_max])
        num_super_y = 1

    cell_width = (map_x_max - map_x_min) / num_super_x if num_super_x > 0 else (map_x_max - map_x_min)
    cell_height = (map_y_max - map_y_min) / num_super_y if num_super_y > 0 else (map_y_max - map_y_min)
    
    actual_local_search_tx_half_width = cell_width / 2.0
    actual_local_search_ty_half_width = cell_height / 2.0
    
    tasks = []
    if live_scan_keypoints_np.shape[0] == 0:
        print("    Warning: Live scan keypoints are empty. Skipping task generation in search_single_level.")
    else:
        for super_cx in super_cx_centers:
            for super_cy in super_cy_centers:
                tasks.append((
                    super_cx, super_cy,
                    live_scan_keypoints_np, global_map_kdtree,
                    actual_local_search_tx_half_width, 
                    actual_local_search_ty_half_width,
                    tx_ty_search_steps_per_cell[0], tx_ty_search_steps_per_cell[1], 
                    theta_candidates_rad, theta_candidates_deg,
                    correspondence_dist_thresh
                ))

    total_super_cells = len(tasks)
    if total_super_cells == 0:
        return [], (super_x_edges, super_y_edges), 0
        
    level_results = [] 
    total_iterations_evaluated_this_level = 0
    current_level_best_score = -1.0

    if num_processes == 0: 
        processed_tasks_count = 0
        for i, task_args in enumerate(tasks):
            result = process_super_grid_cell(task_args) 
            processed_tasks_count +=1
            cell_score, cell_tx, cell_ty, cell_theta, iterations_in_cell, cell_center_coords = result
            total_iterations_evaluated_this_level += iterations_in_cell
            
            if cell_score is not None: 
                level_results.append((cell_score, cell_tx, cell_ty, cell_theta, cell_center_coords[0], cell_center_coords[1]))
                if cell_score > current_level_best_score: current_level_best_score = cell_score
    else: 
        actual_num_processes = multiprocessing.cpu_count() if num_processes is None else num_processes
        results_from_pool = []
        with multiprocessing.Pool(processes=actual_num_processes) as pool:
            processed_tasks_count = 0
            for res in pool.imap_unordered(process_super_grid_cell, tasks):
                results_from_pool.append(res)
                processed_tasks_count +=1
        for result in results_from_pool:
            cell_score, cell_tx, cell_ty, cell_theta, iterations_in_cell, cell_center_coords = result
            total_iterations_evaluated_this_level += iterations_in_cell
            if cell_score is not None: 
                level_results.append((cell_score, cell_tx, cell_ty, cell_theta, cell_center_coords[0], cell_center_coords[1]))
                if cell_score > current_level_best_score: current_level_best_score = cell_score
    
    return level_results, (super_x_edges, super_y_edges), total_iterations_evaluated_this_level


def search_in_super_grids(
    global_map_keypoints_np, live_scan_keypoints_np,
    global_map_x_edges, global_map_y_edges,
    super_grid_cell_size_factor=20.0, local_search_tx_half_width_factor=1.0,
    local_search_ty_half_width_factor=1.0, tx_search_steps=10, ty_search_steps=10,
    search_theta_range_deg=(0, 359), theta_search_steps=36,
    correspondence_distance_threshold_factor=1.5, base_grid_cell_size=1.0,
    num_processes=None
):
    if global_map_keypoints_np.shape[0] == 0: return {'tx':0,'ty':0,'theta_deg':0,'score':-1}, -1, (np.array([]), np.array([])), (np.array([]),np.array([]),np.array([])), 0
    if live_scan_keypoints_np.shape[0] == 0: return {'tx':0,'ty':0,'theta_deg':0,'score':-1}, -1, (np.array([]), np.array([])), (np.array([]),np.array([]),np.array([])), 0

    start_time = time.time()
    current_best_transform_info = {'tx':0,'ty':0,'theta_deg':0,'score':-1}

    all_super_cell_centers_x = []
    all_super_cell_centers_y = []
    all_super_cell_scores = []

    global_map_kdtree = KDTree(global_map_keypoints_np)
    map_x_min, map_x_max = global_map_x_edges[0], global_map_x_edges[-1]
    map_y_min, map_y_max = global_map_y_edges[0], global_map_y_edges[-1]

    actual_super_grid_cell_size = base_grid_cell_size * super_grid_cell_size_factor
    actual_local_search_tx_half_width = actual_super_grid_cell_size * local_search_tx_half_width_factor
    actual_local_search_ty_half_width = actual_super_grid_cell_size * local_search_ty_half_width_factor
    actual_correspondence_dist_thresh = base_grid_cell_size * correspondence_distance_threshold_factor

    super_x_edges = np.arange(map_x_min, map_x_max + actual_super_grid_cell_size, actual_super_grid_cell_size)
    super_y_edges = np.arange(map_y_min, map_y_max + actual_super_grid_cell_size, actual_super_grid_cell_size)
    if len(super_x_edges) <= 1: super_x_edges = np.array([map_x_min, map_x_max])
    if len(super_y_edges) <= 1: super_y_edges = np.array([map_y_min, map_y_max])
    super_cx_centers = (super_x_edges[:-1] + super_x_edges[1:]) / 2
    super_cy_centers = (super_y_edges[:-1] + super_y_edges[1:]) / 2

    theta_candidates_deg = np.linspace(search_theta_range_deg[0], search_theta_range_deg[1], theta_search_steps, endpoint=False)
    theta_candidates_rad = np.deg2rad(theta_candidates_deg)
    
    print("  Transformations will be done on CPU (Numba JIT).")

    tasks = []
    for super_cx in super_cx_centers:
        for super_cy in super_cy_centers:
            tasks.append((
                super_cx, super_cy,
                live_scan_keypoints_np, global_map_kdtree,
                actual_local_search_tx_half_width, actual_local_search_ty_half_width,
                tx_search_steps, ty_search_steps,
                theta_candidates_rad, theta_candidates_deg,
                actual_correspondence_dist_thresh
            ))

    total_super_cells = len(tasks)
    print(f"Super-grid 탐색 시작: {len(super_cx_centers)}x{len(super_cy_centers)} ({total_super_cells}개) Super-cells, {len(theta_candidates_rad)} angles.")
    print(f"  Super-grid cell size: {actual_super_grid_cell_size:.2f}")
    print(f"  Local search half-width (tx,ty): {actual_local_search_tx_half_width:.2f}, {actual_local_search_ty_half_width:.2f}")
    print(f"  Correspondence distance threshold: {actual_correspondence_dist_thresh:.2f}")
    
    total_iterations_evaluated = 0; completed_tasks = 0

    if num_processes == 0:
        print("  Sequential processing for super-grid cells.")
        for i, task_args in enumerate(tasks):
            result = process_super_grid_cell(task_args)
            completed_tasks += 1
            cell_score, cell_tx, cell_ty, cell_theta, iterations_in_cell, cell_center = result
            total_iterations_evaluated += iterations_in_cell
            all_super_cell_centers_x.append(cell_center[0])
            all_super_cell_centers_y.append(cell_center[1])
            all_super_cell_scores.append(cell_score if cell_score > -1 else 0)

            if cell_score > current_best_transform_info['score']:
                current_best_transform_info['score'] = cell_score
                current_best_transform_info['tx'] = cell_tx
                current_best_transform_info['ty'] = cell_ty
                current_best_transform_info['theta_deg'] = cell_theta
                print(f"    New global best: score={current_best_transform_info['score']}, tx={current_best_transform_info['tx']:.2f}, ty={current_best_transform_info['ty']:.2f}, th={current_best_transform_info['theta_deg']:.1f} (from S-Cell ({cell_center[0]:.1f},{cell_center[1]:.1f}))")
    else:
        if num_processes is None: num_processes = multiprocessing.cpu_count()
        print(f"  Parallel processing using {num_processes} processes for super-grid cells.")
        with multiprocessing.Pool(processes=num_processes) as pool:
            results_from_pool = []
            processed_count_in_parallel = 0
            for res in pool.imap_unordered(process_super_grid_cell, tasks):
                results_from_pool.append(res)
                processed_count_in_parallel +=1
                cell_score, cell_tx, cell_ty, cell_theta, iterations_in_cell, cell_center = res
                total_iterations_evaluated += iterations_in_cell
                all_super_cell_centers_x.append(cell_center[0])
                all_super_cell_centers_y.append(cell_center[1])
                all_super_cell_scores.append(cell_score if cell_score > -1 else 0)

                if cell_score > current_best_transform_info['score']:
                    current_best_transform_info['score'] = cell_score
                    current_best_transform_info['tx'] = cell_tx
                    current_best_transform_info['ty'] = cell_ty
                    current_best_transform_info['theta_deg'] = cell_theta

    elapsed_time = time.time() - start_time
    if current_best_transform_info['score'] == -1: print("경고: 유효한 매칭을 찾지 못했습니다. (search_in_super_grids)")
    
    return current_best_transform_info, elapsed_time, \
           (super_x_edges, super_y_edges), \
           (np.array(all_super_cell_centers_x), np.array(all_super_cell_centers_y), np.array(all_super_cell_scores)), \
           total_iterations_evaluated


def hierarchical_adaptive_search(
    global_map_keypoints_np, live_scan_keypoints_np,
    initial_map_x_edges, initial_map_y_edges, 
    level_configs, 
    num_candidates_to_select_per_level, 
    min_candidate_separation_factor, 
    base_grid_cell_size, 
    num_processes
):
    overall_best_score = -1.0 
    overall_best_transform = {'tx': 0.0, 'ty': 0.0, 'theta_deg': 0.0, 'score': -1.0}
    all_levels_viz_data = [] 

    if global_map_keypoints_np.shape[0] == 0 or live_scan_keypoints_np.shape[0] == 0:
        print("Error: Keypoints are empty for hierarchical search.")
        return overall_best_transform, overall_best_score, all_levels_viz_data, 0.0, 0

    global_map_kdtree = KDTree(global_map_keypoints_np)
    total_time_start = time.time()
    grand_total_iterations_evaluated = 0 

    initial_center_x = (initial_map_x_edges[0] + initial_map_x_edges[-1]) / 2
    initial_center_y = (initial_map_y_edges[0] + initial_map_y_edges[-1]) / 2
    processing_candidates_from_prev_level = [
        (-1.0, initial_center_x, initial_center_y, 0.0, initial_center_x, initial_center_y)
    ]
    
    current_level_searched_area_details_for_viz = [] 

    for level_idx, config in enumerate(level_configs):
        print(f"\n=== Processing Level {level_idx + 1} / {len(level_configs)} ===")
        
        all_cell_infos_for_current_level_nms = [] 
        current_level_actual_search_bounds_viz = [] 
        current_level_searched_area_details_for_viz.clear() 

        if not processing_candidates_from_prev_level:
            print(f"  Level {level_idx + 1}: No candidates from previous level to process. Stopping.")
            break

        num_prev_level_candidates = len(processing_candidates_from_prev_level)
        print(f"  Level {level_idx + 1}: To explore {num_prev_level_candidates} candidate region(s) from previous level.")

        for cand_idx, prev_level_cand_info in enumerate(processing_candidates_from_prev_level):
            prev_score, prev_tx, prev_ty, prev_theta, prev_cx, prev_cy = prev_level_cand_info
            
            print(f"  Exploring candidate region {cand_idx + 1}/{num_prev_level_candidates} (based on prev T: tx={prev_tx:.2f}, ty={prev_ty:.2f}, th={prev_theta:.1f})")

            grid_division = config["grid_division"]
            search_area_type = config["search_area_type"]
            area_param = config.get("area_ratio_or_size")

            if level_idx == 0: 
                current_search_x_edges = initial_map_x_edges
                current_search_y_edges = initial_map_y_edges
                center_for_theta_calc = 0.0 
            else:
                center_tx_for_current_search = prev_tx 
                center_ty_for_current_search = prev_ty
                center_for_theta_calc = prev_theta 

                if search_area_type == "relative_to_map" and area_param is not None:
                    map_width_init = initial_map_x_edges[-1] - initial_map_x_edges[0]
                    map_height_init = initial_map_y_edges[-1] - initial_map_y_edges[0]
                    ratio = area_param
                    search_w = map_width_init * np.sqrt(ratio)
                    search_h = map_height_init * np.sqrt(ratio)
                elif search_area_type == "absolute_size" and area_param is not None:
                    search_w, search_h = area_param
                else:
                    print(f"Warning: Invalid search_area_type or area_param for Level {level_idx + 1}. Defaulting to small area around prev_tx, prev_ty.")
                    search_w, search_h = 20.0, 20.0 
                
                current_search_x_edges = np.array([center_tx_for_current_search - search_w / 2, center_tx_for_current_search + search_w / 2])
                current_search_y_edges = np.array([center_ty_for_current_search - search_h / 2, center_ty_for_current_search + search_h / 2])
                
                current_search_x_edges[0] = np.clip(current_search_x_edges[0], initial_map_x_edges[0], initial_map_x_edges[-1]-1e-3)
                current_search_x_edges[1] = np.clip(current_search_x_edges[1], initial_map_x_edges[0]+1e-3, initial_map_x_edges[-1])
                current_search_y_edges[0] = np.clip(current_search_y_edges[0], initial_map_y_edges[0], initial_map_y_edges[-1]-1e-3)
                current_search_y_edges[1] = np.clip(current_search_y_edges[1], initial_map_y_edges[0]+1e-3, initial_map_y_edges[-1])
                if current_search_x_edges[0] >= current_search_x_edges[1] or current_search_y_edges[0] >= current_search_y_edges[1]:
                    print(f"    Skipping candidate region {cand_idx+1} due to invalid search area after clipping.")
                    continue
            
            current_level_actual_search_bounds_viz.append((current_search_x_edges.copy(), current_search_y_edges.copy()))

            theta_steps = config["theta_search_steps"]
            if level_idx == 0 or search_area_type == "full_map":
                theta_min_deg, theta_max_deg = config["theta_range_deg"]
            else:
                theta_center_current = center_for_theta_calc
                theta_min_deg = theta_center_current + config["theta_range_deg_relative"][0]
                theta_max_deg = theta_center_current + config["theta_range_deg_relative"][1]
            
            theta_candidates_deg = np.linspace(theta_min_deg, theta_max_deg, theta_steps, endpoint=False)
            theta_candidates_rad = np.deg2rad(theta_candidates_deg)

            cell_infos_one_search, grid_edges_one_search, iterations_current_call = search_single_level(
                level_idx, global_map_keypoints_np, live_scan_keypoints_np,
                current_search_x_edges, current_search_y_edges, grid_division,
                theta_candidates_rad, theta_candidates_deg, 
                config["correspondence_distance_threshold_factor"] * base_grid_cell_size,
                config["tx_ty_search_steps_per_cell"],
                base_grid_cell_size, global_map_kdtree, num_processes
            )
            grand_total_iterations_evaluated += iterations_current_call 
            
            current_level_searched_area_details_for_viz.append({
                "bounds": (current_search_x_edges.copy(), current_search_y_edges.copy()),
                "grid_edges": grid_edges_one_search, 
                "grid_division": grid_division 
            })

            if cand_idx == 0: 
                representative_super_grid_edges_this_level = grid_edges_one_search

            if cell_infos_one_search:
                all_cell_infos_for_current_level_nms.extend(cell_infos_one_search)
            else:
                print(f"    No results from search_single_level for candidate region {cand_idx + 1}.")

        if not all_cell_infos_for_current_level_nms:
            print(f"  Level {level_idx + 1}: No valid cell information obtained from any candidate region. Stopping.")
            break

        for r_score, r_tx, r_ty, r_theta, _, _ in all_cell_infos_for_current_level_nms:
            if r_score is not None and r_score > overall_best_score:
                overall_best_score = r_score
                overall_best_transform = {'score': r_score, 'tx': r_tx, 'ty': r_ty, 'theta_deg': r_theta}
                print(f"    New overall best (during Level {level_idx+1} processing): score={overall_best_score:.0f}, tx={r_tx:.2f}, ty={r_ty:.2f}, th={r_theta:.1f}")
        
        if current_level_actual_search_bounds_viz: 
            first_search_x_edges, first_search_y_edges = current_level_actual_search_bounds_viz[0]
            nms_cell_size_x = (first_search_x_edges[1] - first_search_x_edges[0]) / config["grid_division"][0] if config["grid_division"][0] > 0 else (first_search_x_edges[1] - first_search_x_edges[0])
            nms_cell_size_y = (first_search_y_edges[1] - first_search_y_edges[0]) / config["grid_division"][1] if config["grid_division"][1] > 0 else (first_search_y_edges[1] - first_search_y_edges[0])
        else: 
            nms_cell_size_x = (initial_map_x_edges[1] - initial_map_x_edges[0]) / config["grid_division"][0] if config["grid_division"][0] > 0 else 1.0
            nms_cell_size_y = (initial_map_y_edges[1] - initial_map_y_edges[0]) / config["grid_division"][1] if config["grid_division"][1] > 0 else 1.0
            print("Warning: NMS cell size calculation fallback due to no search bounds.")

        num_select_for_nms_this_level = num_candidates_to_select_per_level if level_idx < len(level_configs) - 1 else 1
        
        selected_candidates_for_next_level = select_diverse_candidates(
            all_cell_infos_for_current_level_nms, 
            num_select_for_nms_this_level, 
            min_candidate_separation_factor,
            nms_cell_size_x, nms_cell_size_y,
            (initial_map_x_edges[0], initial_map_x_edges[-1]),
            (initial_map_y_edges[0], initial_map_y_edges[-1])
        )

        print(f"  Level {level_idx + 1}: NMS selected {len(selected_candidates_for_next_level)} candidates for L{level_idx+2} (from {len(all_cell_infos_for_current_level_nms)} total cells).")

        if not selected_candidates_for_next_level:
            print(f"  Level {level_idx + 1}: No candidates selected by NMS for further processing. Stopping.")
            break
        
        processing_candidates_from_prev_level = selected_candidates_for_next_level
        
        viz_all_cx = [info[4] for info in all_cell_infos_for_current_level_nms if info[0] is not None and np.isfinite(info[4])]
        viz_all_cy = [info[5] for info in all_cell_infos_for_current_level_nms if info[0] is not None and np.isfinite(info[5])]
        viz_all_scores = [info[0] for info in all_cell_infos_for_current_level_nms if info[0] is not None and np.isfinite(info[4]) and np.isfinite(info[5])]

        level_viz_data = {
            "level": level_idx + 1,
            "all_raw_cell_infos_this_level": all_cell_infos_for_current_level_nms, 
            "searched_areas_details": current_level_searched_area_details_for_viz.copy(), 
            "selected_candidates_after_nms": selected_candidates_for_next_level, 
            "overall_best_at_this_level": overall_best_transform.copy()
        }
        all_levels_viz_data.append(level_viz_data)

        if level_idx == len(level_configs) - 1: 
            print(f"\n--- Hierarchical search finished after {len(level_configs)} levels ---")
            break
            
    total_time_elapsed = time.time() - total_time_start
    print(f"\n--- Hierarchical Adaptive Search Complete ---")

    final_selected_transform = overall_best_transform 
    final_selected_score = overall_best_score

    if processing_candidates_from_prev_level: 
        last_level_best_candidate = processing_candidates_from_prev_level[0] 
        final_selected_score = last_level_best_candidate[0]
        final_selected_transform = {
            'score': last_level_best_candidate[0],
            'tx': last_level_best_candidate[1],
            'ty': last_level_best_candidate[2],
            'theta_deg': last_level_best_candidate[3]
        }
        print(f"  Final Best Transform (from last level NMS): tx={final_selected_transform['tx']:.3f}, ty={final_selected_transform['ty']:.3f}, theta={final_selected_transform['theta_deg']:.2f} deg")
        print(f"  Best Score (from last level NMS): {final_selected_score}")
    elif overall_best_score > -1.000001: 
        print(f"  Warning: No candidate from last level NMS. Using overall best transform.")
        print(f"  Overall Best Transform: tx={overall_best_transform['tx']:.3f}, ty={overall_best_transform['ty']:.3f}, theta={overall_best_transform['theta_deg']:.2f} deg")
        print(f"  Overall Best Score: {overall_best_score}")
    else:
        print("  No valid transformation found throughout the hierarchical search.")
    
    print(f"  Total Hierarchical Search Time: {total_time_elapsed:.2f} 초")
    print(f"  Total Transformations Evaluated in Hierarchy: {grand_total_iterations_evaluated}") 

    return final_selected_transform, final_selected_score, all_levels_viz_data, total_time_elapsed, grand_total_iterations_evaluated