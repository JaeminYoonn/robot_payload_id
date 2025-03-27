import argparse
import logging
import time

from functools import partial
from pathlib import Path

import numpy as np
import wandb
import yaml

from robot_payload_id.environment import create_arm
from robot_payload_id.optimization import (
    CostFunction,
    ExcitationTrajectoryOptimizerBsplineBlackBoxALNumeric,
    ExcitationTrajectoryOptimizerFourierBlackBoxALNumeric,
    ExcitationTrajectoryOptimizerFourierBlackBoxNumeric,
    ExcitationTrajectoryOptimizerFourierBlackBoxSymbolic,
    ExcitationTrajectoryOptimizerFourierBlackBoxSymbolicNumeric,
    ExcitationTrajectoryOptimizerFourierSnopt,
)
from robot_payload_id.utils import FourierSeriesTrajectoryAttributes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_one_link_arm",
        action="store_true",
        help="Use one link arm instead of 7-DOF arm.",
    )
    parser.add_argument(
        "--load_data_matrix",
        action="store_true",
        help="Load data matrix from file instead of computing it.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        required=True,
        choices=["snopt", "black_box", "both"],
        help="Optimizer to use. If both, then use black-box to initialize SNOPT.",
    )
    parser.add_argument(
        "--cost_function",
        type=CostFunction,
        required=True,
        choices=list(CostFunction),
        help="Cost function to use.",
    )
    parser.add_argument(
        "--num_fourier_terms",
        type=int,
        default=5,
        help="Number of Fourier terms to use.",
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=0.3 * np.pi,
        help="Frequency of the Fourier series trajectory. The period of the trajectory "
        + "is 2 * pi / omega. Hence, halfing the period will double the unique Fourier "
        + "series trajectory duration.",
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=1000,
        help="The number of timesteps to use.",
    )
    parser.add_argument(
        "--num_control_points",
        type=int,
        default=10,
        help="The number of control points to use. Only used for B-spline optimization.",
    )
    parser.add_argument(
        "--min_time_horizon",
        type=float,
        default=10,
        help="The time horizon/ duration of the trajectory.",
    )
    parser.add_argument(
        "--max_time_horizon",
        type=float,
        default=10,
        help="The time horizon/ duration of the trajectory. The sampling time step is "
        + "computed as max_time_horizon / num_timesteps.",
    )
    parser.add_argument(
        "--snopt_iteration_limit",
        type=int,
        default=1000,
        help="Iteration limit for SNOPT.",
    )
    parser.add_argument(
        "--max_al_iterations",
        type=int,
        default=10,
        help="Maximum number of augmented Lagrangian iterations. Only used for "
        + "black-box with augmented Lagrangian optimization.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=3000,
        help="Budget for black-box optimization. If '--no_al' is set, then this is the "
        + "number of iterations to run the optimizer for at each augmented Lagrangian "
        + "iteration.",
    )
    parser.add_argument(
        "--mu_initial",
        type=float,
        default=5.0,
        help="Initial value of the augmented Lagrangian parameter. Only used for "
        + "black-box with augmented Lagrangian optimization.",
    )
    parser.add_argument(
        "--mu_multiplier",
        type=float,
        default=1.5,
        help="Multiplier for the augmented Lagrangian parameter. Only used for black-box "
        + "with augmented Lagrangian optimization.",
    )
    parser.add_argument(
        "--mu_max",
        type=float,
        default=1e3,
        help="Maximum value of the augmented Lagrangian parameter. Only used for "
        + "black-box with augmented Lagrangian optimization.",
    )
    parser.add_argument(
        "--use_symbolic_computations",
        action="store_true",
        help="Whether to use symbolic computations. If False, then If False, then the "
        + "data matrix is numerically computed from scratch at each iteration.",
    )
    parser.add_argument(
        "--not_symbolically_reexpress_data_matrix",
        action="store_true",
        help="Whether to not symbolically re-express the data matrix. If True, then the "
        + "data matrix is numerically re-expressed at each iteration. Only used for "
        + "black-box optimization.",
    )
    parser.add_argument(
        "--use_bspline",
        action="store_true",
        help="Whether to use B-spline instead of Fourier series trajectory "
        + "parameterization.",
    )
    parser.add_argument(
        "--no_al",
        action="store_true",
        help="Whether to not use augmented Lagrangian. Only used for black-box "
        + "optimization.",
    )
    parser.add_argument(
        "--nevergrad_method",
        type=str,
        default="NGOpt",
        help="Nevergrad method to use. Only used for black-box optimization.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers to use for parallel optimization. Ignores if the "
        + "optimizer doesn't support parallel optimization.",
    )
    parser.add_argument(
        "--logging_path",
        type=Path,
        default=None,
        help="Path to the directory to save the logs to. Only used for black-box "
        + "optimization.",
    )
    parser.add_argument(
        "--traj_initial",
        type=Path,
        default=None,
        help="Path to the initial trajectory.",
    )
    parser.add_argument(
        "--add_rotor_inertia",
        action="store_true",
        help="Add reflected rotor inertia to the optimization. NOTE: This will lead "
        + "to horrible conditioning and it is recommended to use "
        + "`add_reflected_inertia` instead.",
    )
    parser.add_argument(
        "--add_reflected_inertia",
        action="store_true",
        help="Add reflected inertia to the optimization. NOTE: This is mutually "
        + "exclusive with `add_rotor_inertia`.",
    )
    parser.add_argument(
        "--add_viscous_friction",
        action="store_true",
        help="Add viscous friction to the optimization.",
    )
    parser.add_argument(
        "--add_dynamic_dry_friction",
        action="store_true",
        help="Add dynamic dry friction to the optimization.",
    )
    parser.add_argument(
        "--not_add_endpoint_constraints",
        action="store_true",
        help="Add zero velocity/acceleration endpoint constraints to the optimization. "
        + "Note that it might be possible to achieve better performance by not "
        + "including them and then solving two simple trajopt problems to reach the "
        + "start and end points. However, these might then turn out infeasible.",
    )
    parser.add_argument(
        "--constrain_position_endpoints",
        action="store_true",
        help="Whether to add position constraints at the start and end of the "
        + "trajectory. This is only used if an initial trajectory is provided.",
    )
    parser.add_argument(
        "--payload_only",
        action="store_true",
        help="Only consider the 10 inertial parameters of the last link. These are the "
        + "parameters that we want to estimate for payload identification.",
    )
    parser.add_argument(
        "--no_obstacles",
        action="store_true",
        help="Whether to not include obstacles in the optimization.",
    )
    parser.add_argument(
        "--initial_guess_scaling",
        type=float,
        default=1.0,
        help="The scaling factor to use for the initial guess. The initial guess is "
        "randomly generated between -0.5 and 0.5 and then multiplied by this scaling "
        "factor.",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["disabled", "online", "offline"],
        help="WandB mode.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Log level.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    wandb.init(
        project="robot_payload_id",
        name=f"optimal_trajectory_design {time.strftime('%Y-%m-%d-%H-%M-%S')}",
        config=vars(args),
        mode=args.wandb_mode,
    )

    logging_path = args.logging_path
    if logging_path is not None:
        logging_path.mkdir(parents=True)
        args_dict = vars(args)
        if wandb.run is not None:
            args_dict["wandb_url"] = wandb.run.get_url()
        yaml.dump(args_dict, open(logging_path / "args.yaml", "w"))

    use_one_link_arm = args.use_one_link_arm
    num_joints = 1 if use_one_link_arm else 7

    data_matrix_dir_path, model_path = None, None
    if args.load_data_matrix:
        data_matrix_dir_path = Path(
            "data/symbolic_data_matrix_one_link_arm"
            if use_one_link_arm
            else "data/symbolic_data_matrix_iiwa"
        )
    model_path = (
        "./models/one_link_arm_with_obstacle.dmd.yaml"
        if use_one_link_arm
        else (
            "./models/panda.dmd.yaml"
            if args.no_obstacles
            else "./models/panda_with_obstacles.dmd.yaml"
        )
    )

    arm_components = create_arm(arm_file_path=model_path, num_joints=num_joints)
    plant = arm_components.plant
    plant_context = plant.GetMyContextFromRoot(
        arm_components.diagram.CreateDefaultContext()
    )
    robot_model_instance_name = "arm" if use_one_link_arm else "panda"
    robot_model_instance_idx = plant.GetModelInstanceByName(robot_model_instance_name)

    optimizer = args.optimizer
    cost_function = args.cost_function
    num_fourier_terms = args.num_fourier_terms
    omega = args.omega
    num_timesteps = args.num_timesteps
    min_time_horizon = args.min_time_horizon
    max_time_horizon = args.max_time_horizon
    snopt_iteration_limit = args.snopt_iteration_limit
    budget = args.budget
    max_al_iterations = args.max_al_iterations
    mu_initial = args.mu_initial
    mu_multiplier = args.mu_multiplier
    mu_max = args.mu_max
    nevergrad_method = args.nevergrad_method
    num_workers = args.num_workers
    num_control_points = args.num_control_points
    traj_initial = args.traj_initial
    add_rotor_inertia = args.add_rotor_inertia
    add_reflected_inertia = args.add_reflected_inertia
    add_viscous_friction = args.add_viscous_friction
    add_dynamic_dry_friction = args.add_dynamic_dry_friction
    add_endpoint_constraints = not args.not_add_endpoint_constraints
    constrain_position_endpoints = args.constrain_position_endpoints
    payload_only = args.payload_only
    initial_guess_scaling = args.initial_guess_scaling

    # Set seed.
    np.random.seed(42)

    if args.use_bspline:
        assert (
            optimizer == "black_box"
        ), "B-spline is only supported for black-box optimization."
        assert (
            not args.no_al
        ), "Augmented Lagrangian is required for B-spline optimization."

        black_box_optimizer = (
            ExcitationTrajectoryOptimizerBsplineBlackBoxALNumeric(
                num_joints=num_joints,
                cost_function=cost_function,
                plant=plant,
                plant_context=plant_context,
                robot_model_instance_idx=robot_model_instance_idx,
                model_path=model_path,
                num_timesteps=num_timesteps,
                num_control_points=num_control_points,
                min_trajectory_duration=min_time_horizon,
                max_trajectory_duration=max_time_horizon,
                max_al_iterations=max_al_iterations,
                budget_per_iteration=budget,
                mu_initial=mu_initial,
                mu_multiplier=mu_multiplier,
                mu_max=mu_max,
                add_rotor_inertia=add_rotor_inertia,
                add_reflected_inertia=add_reflected_inertia,
                add_viscous_friction=add_viscous_friction,
                add_dynamic_dry_friction=add_dynamic_dry_friction,
                payload_only=payload_only,
                include_endpoint_constraints=add_endpoint_constraints,
                constrain_position_endpoints=constrain_position_endpoints,
                nevergrad_method=nevergrad_method,
                spline_order=4,
                traj_initial=traj_initial,
                logging_path=logging_path,
            )
            if num_workers == 1
            else partial(
                ExcitationTrajectoryOptimizerBsplineBlackBoxALNumeric.optimize_parallel,
                num_joints=num_joints,
                cost_function=cost_function,
                model_path=model_path,
                robot_model_instance_name=robot_model_instance_name,
                num_timesteps=num_timesteps,
                num_control_points=num_control_points,
                min_trajectory_duration=min_time_horizon,
                max_trajectory_duration=max_time_horizon,
                max_al_iterations=max_al_iterations,
                budget_per_iteration=budget,
                mu_initial=mu_initial,
                mu_multiplier=mu_multiplier,
                mu_max=mu_max,
                add_rotor_inertia=add_rotor_inertia,
                add_reflected_inertia=add_reflected_inertia,
                add_viscous_friction=add_viscous_friction,
                add_dynamic_dry_friction=add_dynamic_dry_friction,
                payload_only=payload_only,
                include_endpoint_constraints=add_endpoint_constraints,
                constrain_position_endpoints=constrain_position_endpoints,
                nevergrad_method=nevergrad_method,
                spline_order=4,
                traj_initial=traj_initial,
                num_workers=num_workers,
                logging_path=logging_path,
            )
        )
    else:
        # Create the black-box optimizer
        if optimizer != "snopt":
            if not args.no_al:
                black_box_optimizer = (
                    (
                        ExcitationTrajectoryOptimizerFourierBlackBoxALNumeric(
                            num_joints=num_joints,
                            cost_function=cost_function,
                            num_fourier_terms=num_fourier_terms,
                            omega=omega,
                            num_timesteps=num_timesteps,
                            time_horizon=max_time_horizon,
                            plant=plant,
                            plant_context=plant_context,
                            robot_model_instance_idx=robot_model_instance_idx,
                            max_al_iterations=max_al_iterations,
                            budget_per_iteration=budget,
                            mu_initial=mu_initial,
                            mu_multiplier=mu_multiplier,
                            mu_max=mu_max,
                            model_path=model_path,
                            add_rotor_inertia=add_rotor_inertia,
                            add_reflected_inertia=add_reflected_inertia,
                            add_viscous_friction=add_viscous_friction,
                            add_dynamic_dry_friction=add_dynamic_dry_friction,
                            payload_only=payload_only,
                            include_endpoint_constraints=add_endpoint_constraints,
                            nevergrad_method=nevergrad_method,
                            traj_initial=traj_initial,
                            initial_guess_scaling=initial_guess_scaling,
                            logging_path=logging_path,
                        )
                    )
                    if num_workers == 1
                    else partial(
                        ExcitationTrajectoryOptimizerFourierBlackBoxALNumeric.optimize_parallel,
                        num_joints=num_joints,
                        cost_function=cost_function,
                        num_fourier_terms=num_fourier_terms,
                        omega=omega,
                        num_timesteps=num_timesteps,
                        time_horizon=max_time_horizon,
                        max_al_iterations=max_al_iterations,
                        budget_per_iteration=budget,
                        mu_initial=mu_initial,
                        mu_multiplier=mu_multiplier,
                        mu_max=mu_max,
                        model_path=model_path,
                        robot_model_instance_name=robot_model_instance_name,
                        num_workers=num_workers,
                        add_rotor_inertia=add_rotor_inertia,
                        add_reflected_inertia=add_reflected_inertia,
                        add_viscous_friction=add_viscous_friction,
                        add_dynamic_dry_friction=add_dynamic_dry_friction,
                        payload_only=payload_only,
                        include_endpoint_constraints=add_endpoint_constraints,
                        nevergrad_method=nevergrad_method,
                        traj_initial=traj_initial,
                        initial_guess_scaling=initial_guess_scaling,
                        logging_path=logging_path,
                    )
                )
            elif (
                args.use_symbolic_computations
                and not args.not_symbolically_reexpress_data_matrix
            ):
                black_box_optimizer = (
                    ExcitationTrajectoryOptimizerFourierBlackBoxSymbolic(
                        num_joints=num_joints,
                        cost_function=cost_function,
                        num_fourier_terms=num_fourier_terms,
                        omega=omega,
                        num_timesteps=num_timesteps,
                        time_horizon=max_time_horizon,
                        plant=plant,
                        robot_model_instance_idx=robot_model_instance_idx,
                        budget=budget,
                        nevergrad_method=nevergrad_method,
                        traj_initial=traj_initial,
                        logging_path=logging_path,
                        data_matrix_dir_path=data_matrix_dir_path,
                        model_path=model_path,
                    )
                )
            elif (
                args.use_symbolic_computations
                and args.not_symbolically_reexpress_data_matrix
            ):
                black_box_optimizer = (
                    ExcitationTrajectoryOptimizerFourierBlackBoxSymbolicNumeric(
                        num_joints=num_joints,
                        cost_function=cost_function,
                        num_fourier_terms=num_fourier_terms,
                        omega=omega,
                        num_timesteps=num_timesteps,
                        time_horizon=max_time_horizon,
                        plant=plant,
                        robot_model_instance_idx=robot_model_instance_idx,
                        budget=budget,
                        nevergrad_method=nevergrad_method,
                        logging_path=logging_path,
                        data_matrix_dir_path=data_matrix_dir_path,
                        model_path=model_path,
                    )
                )
            else:
                black_box_optimizer = (
                    ExcitationTrajectoryOptimizerFourierBlackBoxNumeric(
                        num_joints=num_joints,
                        cost_function=cost_function,
                        num_fourier_terms=num_fourier_terms,
                        omega=omega,
                        num_timesteps=num_timesteps,
                        time_horizon=max_time_horizon,
                        plant=plant,
                        robot_model_instance_idx=robot_model_instance_idx,
                        budget=budget,
                        add_rotor_inertia=add_rotor_inertia,
                        add_reflected_inertia=add_reflected_inertia,
                        add_viscous_friction=add_viscous_friction,
                        add_dynamic_dry_friction=add_dynamic_dry_friction,
                        payload_only=payload_only,
                        nevergrad_method=nevergrad_method,
                        traj_initial=traj_initial,
                        initial_guess_scaling=initial_guess_scaling,
                        logging_path=logging_path,
                        model_path=model_path,
                    )
                )
        # Create the SNOPT optimizer
        if optimizer != "black_box":
            snopt_optimizer = ExcitationTrajectoryOptimizerFourierSnopt(
                data_matrix_dir_path=data_matrix_dir_path,
                model_path=model_path,
                num_joints=num_joints,
                cost_function=cost_function,
                num_fourier_terms=num_fourier_terms,
                omega=omega,
                num_timesteps=num_timesteps,
                time_horizon=max_time_horizon,
                plant=plant,
                robot_model_instance_idx=robot_model_instance_idx,
                iteration_limit=snopt_iteration_limit,
            )

    logging.info("Starting optimization")
    if optimizer == "black_box":
        if num_workers > 1 and callable(black_box_optimizer):
            black_box_optimizer()
        else:
            black_box_optimizer.optimize()
    elif optimizer == "snopt":
        snopt_optimizer.set_initial_guess(
            a=np.random.rand(num_joints * num_fourier_terms) - 0.5,
            b=np.random.rand(num_joints * num_fourier_terms) - 0.5,
            q0=np.random.rand(num_joints) - 0.5,
        )
        snopt_optimizer.optimize()
    else:
        traj_attrs: FourierSeriesTrajectoryAttributes = black_box_optimizer.optimize()
        snopt_optimizer.set_initial_guess(
            traj_attrs.a_values, traj_attrs.b_values, traj_attrs.q0_values
        )
        snopt_optimizer.optimize()


if __name__ == "__main__":
    main()
