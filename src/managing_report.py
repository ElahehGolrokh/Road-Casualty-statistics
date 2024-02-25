import json
import os


class ReportManager:
    """
    This class reads, updates and then saves the updated json file

    ...
    Attributes
    ----------
        author_name: author_name which is going to be used as file
                     name for final json file

    Private Methods
    ---------------
        _create_json()
        _read_json()

    Public Methods
    --------------
        write_json()
        update_json()
    """
    def __init__(self) -> None:
        self._create_json()

    def _create_json(self) -> None:
        """
        Creates a json file if it is not already there. This file
        is going to save all evaluation results of a specific run
        """
        run_dir = 'run/report'
        if not os.path.exists(run_dir):
            os.mkdir(run_dir)
        self.file_path = os.path.join(run_dir,
                                      f'run_report.json')
        if not os.path.exists(self.file_path):
            os.mknod(self.file_path)
            self.write_json(report=[])

    def _read_json(self) -> list:
        """
        Reads a json file

        Returns:
            report: a list of reports stored in the json file
        """
        with open(self.file_path, "r") as json_file:
            report = json.load(json_file)
        return report

    def update_json(self, report_dict: dict) -> None:
        """
        Reads json file containing reports of previouse pages, then
        updates report with the report of new run and saves it
        """
        list_of_runs = self._read_json()
        list_of_runs.append(report_dict)
        self.write_json(list_of_runs)

    def write_json(self, report: list) -> None:
        """
        Writes the report list as a json file

        Args:
            report: a list containing reports of each model charecteristics
                    and evaluation results
        """
        with open(self.file_path, 'w') as file:
            json.dump(report, file, indent=4)
        file.close()