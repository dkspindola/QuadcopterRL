from CurriculumConfigs import env_config0, env_config1, env_config2, env_config3, env_config4


def curriculum_fn(train_results, task_settable_env, env_ctx):
    env_configs = [env_config0, env_config1, env_config2, env_config3, env_config4]
    current_task = task_settable_env.get_task()
    new_task = current_task

    entropy = ((((train_results["info"])["learner"])["default_policy"])["learner_stats"])["entropy"]

    if entropy <= 0:
        new_task = env_configs[current_task["level"] + 1]

    return new_task
