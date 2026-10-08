def _default_env(name):
    if name != "nous":
        raise NotImplementedError(f"Tool environment {name} is not implemented")

    from casd.tool.envs.nous import NousToolEnv

    return NousToolEnv
