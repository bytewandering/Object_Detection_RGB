import copy
from .training_tasks.TrainingTask_Object_Detection_RGB import TrainingTask_Object_Detection_RGB
from .training_tasks.TrainingTask_FOMO_RGB import TrainingTask_FOMO_RGB
from .testing_tasks.TestingTask_Object_Detection_RGB import TestingTask_Object_Detection_RGB
from .testing_tasks.TestingTask_FOMO_RGB import TestingTask_FOMO_RGB


def build_task(config):
    training_task_config = copy.deepcopy(config)
    task_name = training_task_config.pop("task_name")
    if task_name == "TrainingTask_Object_Detection_RGB":
        task = TrainingTask_Object_Detection_RGB(cfg=training_task_config)
    elif task_name == "TrainingTask_FOMO_RGB":
        task = TrainingTask_FOMO_RGB(cfg=training_task_config)
    else:
        raise NotImplementedError

    return task


def build_test_task(config, checkpoint, **options):
    """The test task matching the training task_name saved in the run's config.yml."""
    testing_task_config = copy.deepcopy(config)
    task_name = testing_task_config.pop("task_name")
    if task_name == "TrainingTask_Object_Detection_RGB":
        task = TestingTask_Object_Detection_RGB(cfg=testing_task_config, checkpoint=checkpoint, **options)
    elif task_name == "TrainingTask_FOMO_RGB":
        task = TestingTask_FOMO_RGB(cfg=testing_task_config, checkpoint=checkpoint, **options)
    else:
        raise NotImplementedError

    return task
