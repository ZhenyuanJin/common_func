import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import common_functions as cf
import tqdm

def test_experiment():
    '''
    用于测试experiment的各种功能是否正常
    '''
    class FitODETrainer(cf.Trainer):
        def __init__(self):
            super().__init__()

        def _set_required_key_list(self):
            self.required_key_list = ['ode_solver1']

        def _set_optional_key_value_dict(self):
            self.optional_key_value_dict = {
                'ode_solver2': 'rk4',
                'ode_solver3': 'rk4',
            }

        def add_info_to_info_container(self):
            self.info_container.set_value(1, self.params['ode_solver1'])

        def _set_name(self):
            self.name = 'trainer'

        def _config_data_keeper(self):
            self.data_keeper_name = self.name
            self.data_keeper_kwargs = {'data_type': 'dict', 'save_load_method': 'separate'}

        def run_detail(self):
            self.data_keeper.data['0'] = 1


    class FitODESimulator(cf.Simulator):
        def __init__(self):
            super().__init__()

        def _set_required_key_list(self):
            self.required_key_list = ['ode_solver1']

        def _set_optional_key_value_dict(self):
            self.optional_key_value_dict = {
                'ode_solver2': 'rk4',
                'ode_solver3': 'rk4',
            }

        def _set_name(self):
            self.name = 'simulator'

        def _config_data_keeper(self):
            self.data_keeper_name = self.name
            self.data_keeper_kwargs = {'data_type': 'OrderedDataContainer', 'save_load_method': 'lmdb', 'param_order': ['ode_solver1', 'ode_solver2', 'ode_solver3']}

        def run_detail(self):
            self.data_keeper.data['0'] = 2
            assert self.trainer_data_keeper.get_value('0') == 1


    class FitODEAnalyzer(cf.Analyzer):
        def __init__(self):
            super().__init__()

        def _set_required_key_list(self):
            self.required_key_list = ['ode_solver1']

        def _set_optional_key_value_dict(self):
            self.optional_key_value_dict = {
                'ode_solver2': 'rk4',
                'ode_solver3': 'rk4',
            }

        def _set_name(self):
            self.name = 'analyzer'

        def _config_data_keeper(self):
            self.data_keeper_name = self.name
            self.data_keeper_kwargs = {'data_type': 'dict', 'save_load_method': 'separate'}

        def run_detail(self):
            self.data_keeper.data['0'] = 3
            assert self.trainer_data_keeper.get_value('0') == 1
            assert self.simulator_data_keeper.get_value('0') == 2


    class FitODE(cf.Experiment):
        def __init__(self):
            super().__init__()

        def _set_name(self):
            self.name = 'fit_ode'

        def _minimal_init_tools(self):
            trainer = FitODETrainer()
            simulator = FitODESimulator()
            analyzer = FitODEAnalyzer()
            self.tools = [trainer, simulator, analyzer]


    class FitODE_new(FitODE):
        def _set_name(self):
            self.name = 'fit_ode_new'

        def run_detail(self, **kwargs):
            assert self.fit_ode_data_keeper_dict['trainer'].get_value('0') == 1
            assert self.fit_ode_data_keeper_dict['simulator'].get_value('0') == 2
            assert self.fit_ode_data_keeper_dict['analyzer'].get_value('0') == 3
            for i in tqdm.tqdm([0, 1, 2]):
                pass
            super().run_detail(**kwargs)


    class ComposeFITODE(cf.ComposedExperiment):
        def _minimal_init_experiments(self):
            fitODE1 = FitODE()
            fitODE2 = FitODE_new()
            self.experiments = [fitODE1, fitODE2]

    tool_params_dict = {
        'trainer': {
            'ode_solver1': 'rk4',
        },
        'simulator': {
            'ode_solver1': 'rk2',
            'ode_solver2': 'rk2',
            'ode_solver3': 'rk2',
        },
        'analyzer': {
            'ode_solver1': 'rk3',
            'ode_solver2': 'rk3',
            'ode_solver3': 'rk3',
        }
    }

    tool_config_dict = {
    }

    fitODE = FitODE()
    fitODE.set_tool_params_dict(tool_params_dict)
    fitODE.set_tool_config_dict(tool_config_dict)
    fitODE.set_basedir('./test_ode1')
    fitODE.set_code_file_list([])
    fitODE.run()
    assert fitODE.trainer.data_keeper.get_value('0') == 1
    assert fitODE.simulator.data_keeper.get_value('0') == 2
    assert fitODE.analyzer.data_keeper.get_value('0') == 3

    fitODE = FitODE()
    fitODE.set_tool_params_dict(tool_params_dict)
    fitODE.set_basedir('./test_ode1')
    fitODE.set_current_time('best1')
    fitODE.run()
    assert fitODE.trainer.data_keeper.get_value('0') == 1
    assert fitODE.simulator.data_keeper.get_value('0') == 2
    assert fitODE.analyzer.data_keeper.get_value('0') == 3

    fitODE = FitODE()
    fitODE.load('./test_ode1/best1')
    assert fitODE.trainer.data_keeper.get_value('0') == 1
    assert fitODE.simulator.data_keeper.get_value('0') == 2
    assert fitODE.analyzer.data_keeper.get_value('0') == 3

    fitODE = ComposeFITODE()
    fitODE.set_experiment_params_dict({'fit_ode': tool_params_dict, 
                                    'fit_ode_new': tool_params_dict})
    fitODE.set_basedir('./test_ode_compose')
    fitODE.set_code_file_list([])
    fitODE.run()
    assert fitODE.fit_ode.trainer.data_keeper.get_value('0') == 1
    assert fitODE.fit_ode.simulator.data_keeper.get_value('0') == 2
    assert fitODE.fit_ode.analyzer.data_keeper.get_value('0') == 3