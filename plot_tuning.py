import optuna
from pprint import pprint

from optuna.visualization import  plot_param_importances, plot_contour, plot_slice

if __name__ == "__main__":
    try:
        storage = optuna.storages.RDBStorage(
            url="sqlite:///a3c_tetris_hpo.db",
            engine_kwargs={"connect_args": {"timeout": 30}}
        )
        study = optuna.create_study(
            study_name="a3c-tetris",
            storage=storage,
            load_if_exists=True,
            directions=['maximize', 'maximize', 'maximize']
        )
        study.set_metric_names(["Mean rewards", "Mean blocks placed", "Mean lines cleared"])

        param_importance = optuna.importance.get_param_importances(study, target=lambda t: t.values[0])
        pprint(param_importance)

        fig = plot_param_importances(study)
        fig.show()

        countour_reward = plot_contour(study, target=lambda t: t.values[0], target_name="Mean rewards")
        countour_reward.show()

        contour_blocks = plot_contour(study, target=lambda t: t.values[1], target_name="Mean blocks placed")
        contour_blocks.show()

        contour_lines = plot_contour(study, target=lambda t: t.values[2], target_name="Mean lines cleared")
        contour_lines.show()

        slice_reward = plot_slice(study, target=lambda t: t.values[0], target_name="Mean rewards")
        slice_reward.show()

        slice_blocks = plot_slice(study, target=lambda t: t.values[1], target_name="Mean blocks placed")
        slice_blocks.show()

        slice_lines = plot_slice(study, target=lambda t: t.values[2], target_name="Mean lines cleared")
        slice_lines.show()


    except (KeyboardInterrupt, optuna.exceptions.OptunaError) as e:
        print(f"Error: {e}")