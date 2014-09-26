from sklearn import grid_search
from sklearn import cross_validation
from sklearn import pipeline as skpipeline
from sklearn import svm
from sklearn import ensemble
import numpy as np
import matplotlib.pyplot as plt

class Framework(object):
    """Framework for running machine learning tests. This class should not be used. Instead, use another class that inherits from this one.
    The flow for any inheriting class is as such:
    1. Set the pipeline and params. Can do this through either set_by_algos, set_param_grid, or set_pipeline
    2. cvs = run(X, y, [test_X, test_y])
    3. print results with print_results(cvs)
    4. see heatmap of results with grid_heatmpa(predictor, x_range, y_range)
    """

    default_param_grid = {
        'rf':{'rf__n_estimators':[10,20,30], 'rf__criterion':['gini', 'entropy'],
              'rf__max_features':[5,10,15], 'rf__max_depth':['auto', 'log2', 'sqrt', None]},
        'adaboost':{},
        'gradboost':{}
        }

    def __init__(self):
        self.param_grid = None  # Dict of param_grids, where key is name, e.g. rf or svm
        self.pipeline = None # Dict of pipelines, " "
        self.results = None # Dict of results, " "

    def run(self, X, y, test_X=None, test_y=None):
        """
        Returns a dict of algo type to trained, cved, algos
        @X, @y, @test_X, @test_y: All reg arrays and not np arrays
        """
        if not self._check_validity(X, y, must_exist=True):
            print 'Data inputs, X and y, are not valid.'
            return
        elif not self._check_validity(test_X, test_y):
            print 'Data inputs, test_X and test_y, are not valid.'
            return

        if not test_X or not test_y:
            X, y, test_X, test_y = self.split_set(X, y, make_np=True)

        pipelines = self.get_pipeline()
        param_grids = self.get_param_grid()
        cvs = {}

        for key, pipeline in pipelines.iteritems():
            param_grid = param_grids.get(key)
            if not param_grid:
                print 'Skipping %s because we have no params for it.' % key
                continue
            cvs[key] = self._train_grid(pipeline, param_grid, X, y)
        return cvs

    def get_pipeline(self):
        if not self.pipeline:
            self.set_pipeline()
        return self.pipeline

    def get_param_grid(self):
        if not self.param_grid:
            self.set_param_grid()
        return self.param_grid

    @staticmethod
    def make_pipeline(pipe):
        return skpipeline.Pipeline(pipe)

    def set_pipeline(self, pipeline=None):
        """@pipeline: dict from algo name to pipeline"""
        self.pipeline = pipeline or {key:self.make_pipeline([(key, algo())]) for key, algo in self.algorithm_dict.iteritems()}

    def set_param_grid(self, param_grid=None):
        """@param_grid: dict from algo name to dict of kw to range of values"""
        self.param_grid = param_grid or self.default_param_grid

    def set_by_algos(self, *args):
        """@args: list of algo args. If the arg is supported ('svc', 'rf',...), then makes default pipeline and default param_grid with just those"""
        self.param_grid = {arg:self.default_param_grid[arg] for arg in args if arg in self.default_param_grid}
        self.set_pipeline()
        bad_keys = [key for key in self.pipeline.keys() if key not in args]
        for key in bad_keys:
            del self.pipeline[key]

    def split_set(self, X, y, test_size=.5, make_np=True):
        train_X, test_X, train_y, test_y = cross_validation.train_test_split(X, y, test_size=test_size)
        if make_np:
            return self._convert_to_np(train_X, train_y, test_X, test_y)
        return train_X, train_y, test_X, test_y

    @staticmethod
    def _convert_to_np(*args):
        return tuple(np.array(arg) for arg in args)

    @staticmethod
    def _train_grid(pipeline, param_grid, X, y):
        cv = grid_search.GridSearchCV(
            pipeline, param_grid, verbose=2, n_jobs=-1)
        cv.fit(X, y)
        return cv

    @staticmethod
    def _check_validity(X, y, must_exist=False):
        if must_exist and (X == None or y == None):
            return False
        elif X == None and y == None:
            return True
        elif X == None or y == None:
            return False
        elif len(X) != len(y):
            return False
        return True

    @classmethod
    def print_results(cls, cvs):
        """@cvs: a dict of algo name to predictor"""
        for name, predictor in cvs.iteritems():
            print name
            print predictor.best_score_
            print predictor.best_params_
            print predictor.best_estimator_
            print predictor.grid_scores_
            print '\n'

    @classmethod
    def grid_heatmap(cls, predictor, x_range, y_range):
        """
        @predictor: a single predictor
        @x_range: a range of values for the x axis
        @y_range: a range of values for the y axis
        """
        plt.figure(figsize=(8, 6))
        scores = predictor.grid_scores_
        scores = [x[1] for x in score_dict]
        scores = np.array(scores).reshape(len(x_range), len(y_range))
        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.95)
        plt.imshow(scores, interpolation='nearest', cmap=plt.cm.spectral)
        plt.xlabel('gamma')
        plt.ylabel('C')
        plt.colorbar()
        plt.xticks(np.arange(len(y_range)), y_range, rotation=45)
        plt.yticks(np.arange(len(x_range)), x_range)
        plt.show()

class Classifier(Framework):
    algorithm_dict = {
        'svc':svm.SVC, 'rf':ensemble.RandomForestClassifier,
        'adaboost':ensemble.AdaBoostClassifier, 'gradboost':ensemble.GradientBoostingClassifier
        }
    def __init__(self):
        Framework.__init__(self)
        self.default_param_grid['svc'] = {'svc__C':10.0 ** np.arange(-2, 9), 'svc__gamma':10.0 ** np.arange(-5, 4)}

class Regressor(Framework):
    algorithm_dict = {
        'svc':svm.SVR, 'rf':ensemble.RandomForestRegressor,
        'adaboost':ensemble.AdaBoostRegressor, 'gradboost':ensemble.GradientBoostingRegressor
        }
    def __init__(self):
        Framework.__init__(self)
        self.default_param_grid['svr'] = {'C':10.0 ** np.arange(-2, 9), 'gamma':10.0 ** np.arange(-5, 4)}
