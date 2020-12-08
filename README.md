# football-imitation-learning

Application of imitation learning to football (soccer) video game.

## Usage

### Setup

First, navigate to the football directory with `cd football`.

To install the required dependencies and build the environment, run

```
pip3 install -r requirements.txt
pip3 install .
```

### Episode Data Collection

To execute the AI in the environment and collect episode data, run `python3 execute_ai.py SCENARIO REPRESENTATION ACTION_DATA_FILE_NAME >> ACTION_DATA_FILE_NAME'`. where `SCENARIO` is the scenario the AI is executing in, `REPRESENTATION` is the observation type (`raw`, `pixels`, etc.), and where `ACTION_DATA_FILE_NAME` is the file to which action data will be written to by redirecting STDOUT.

The episodes (states and corresponding actions) for this execution will be stored at `episodes/episodes_SCENARIO_TS.pkl`, respectively, where `TS` is the ending timestamp of the execution in seconds from epoch time. The file to which action data was written to, denoted by `ACTION_DATA_FILE_NAME`, can be deleted.

The following is an example of how to execute the AI in the environment and collect episode data.

```
python3 execute_ai.py 11_vs_11_stochastic raw ad.csv >> ad.csv
```

The episodes for this execution will be stored at `episodes/episodes_11_vs_11_stochastic_TS.pkl`, where `TS` is the ending timestamp of the execution in seconds from epoch time.

### Training and Validation

To run training and validation, run

```
python3 train.py --env_name ENV_NAME
```

where `ENV_NAME` is the name of the environment/scenario (e.g. `academy_empty_goal_close`).

The number of validation episodes, max length of each test episode, directory with the expert data, number of epochs, and batch size can be set using the flags `num_episodes_val`, `episode_len`, `data_dir`, `num_epochs`, and `batch_size`, respectively.

## Notes

At the moment, the assumption is made that the agent will control the left team.

If `pixels` or `pixels_gray` is selected as the observation type, the execution will be rendered and displayed.

## References

We are using the [Google Football Research](https://ai.googleblog.com/2019/06/introducing-google-research-football.html) environment.

