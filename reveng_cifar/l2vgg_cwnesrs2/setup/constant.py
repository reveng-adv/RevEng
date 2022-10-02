import os


class Paths(object):
    """
    Contains paths to directories used in the analysis.

    The user should modify data_dir; everything else should work from there.
    """
    def __init__(self):
        
        self.work_dir = '/home/xiawei/rev_adv-main'
        
        self.feat_dir = os.path.join(self.work_dir, 'features')

        self.model_dir = os.path.join(self.work_dir, 'models')

        # where to save results of analyses
        self.results_dir = os.path.join(self.work_dir, 'results')
        

    def make_directories(self):
        """
        Creates the top level data directories.
        """
        os.makedirs(self.feat_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
