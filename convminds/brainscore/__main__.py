import fire

from convminds.brainscore import score as _score_function


def score(model, benchmark):
    result = _score_function(model, benchmark)
    print(result)


if __name__ == '__main__':
    fire.Fire()
