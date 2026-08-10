class Alpha:
    def execute(self) -> str:
        return "alpha"


class Beta:
    def execute(self) -> str:
        return "beta"


def run(worker: Beta) -> str:
    return worker.execute()


def unknown(worker) -> str:
    return worker.execute()
