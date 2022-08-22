from dynaconf import Dynaconf

settings = Dynaconf(
    ROOT_PATH_FOR_DYNACONF="/home/worker/workspace",
    envvar_prefix="DYNACONF",
    settings_files=["settings.toml", ".secrets.toml", "params.yaml"],
)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
