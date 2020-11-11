# football-imitation-learning

Application of imitation learning to football (soccer) video game.

## Usage

First, navigate to the football directory with `cd football`.

To install the required dependencies and build the environment, run `pip3 install .`.

To execute the AI in the environment, run `python3 execute_ai.py >> episodes/actions_SCENARIO_ID.csv`. where `SCENARIO` is the scenario the AI is executing in and where `ID` is the identifier for this execution.

The observations and action data for this execution will be stored at `episodes/observations_SCENARIO_TS.csv` and `episodes/actions_SCENARIO_ID.csv`, respectively, where `TS` is the ending timestamp of the execution in seconds from epoch time.

## References

We are using the [Google Football Research](https://ai.googleblog.com/2019/06/introducing-google-research-football.html) environment.

