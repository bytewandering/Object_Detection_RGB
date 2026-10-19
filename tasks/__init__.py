import copy
from .training_tasks.TrainingTask_NanoDet_RGB import TrainingTask_NanoDet_RGB
from .training_tasks.TrainingTask_FOMO_RGB import TrainingTask_FOMO_RGB
from .testing_tasks.TestingTask_NanoDet_RGB import TestingTask_NanoDet_RGB
from .testing_tasks.TestingTask_FOMO_RGB import TestingTask_FOMO_RGB


# Checkpoints saved before the rename still carry the old task_name.
NANODET_TASK_NAMES = ("TrainingTask_NanoDet_RGB", "TrainingTask_Object_Detection_RGB")


def build_task(config):
    training_task_config = copy.deepcopy(config)
    task_name = training_task_config.pop("task_name")
    if task_name in NANODET_TASK_NAMES:
        task = TrainingTask_NanoDet_RGB(cfg=training_task_config)
    elif task_name == "TrainingTask_FOMO_RGB":
        task = TrainingTask_FOMO_RGB(cfg=training_task_config)
    else:
        raise NotImplementedError

    return task


def build_test_task(config, checkpoint, **options):
    """The test task matching the training task_name saved in the run's config.yml."""
    testing_task_config = copy.deepcopy(config)
    task_name = testing_task_config.pop("task_name")
    if task_name in NANODET_TASK_NAMES:
        task = TestingTask_NanoDet_RGB(cfg=testing_task_config, checkpoint=checkpoint, **options)
    elif task_name == "TrainingTask_FOMO_RGB":
        task = TestingTask_FOMO_RGB(cfg=testing_task_config, checkpoint=checkpoint, **options)
    else:
        raise NotImplementedError

    return task
