% File: notebooks/scripts_for_notebook_04/script_zero.m
% Utility script for initialization and helper functions

fprintf('   --> Executing script_zero.m\n');

% Initialize common variables if not already defined
if ~exist('global_param', 'var')
    global_param = 10;
end

% Set default plotting parameters
set(0, 'DefaultFigurePosition', [100, 100, 800, 600]);
set(0, 'DefaultAxesFontSize', 12);
set(0, 'DefaultLineLineWidth', 1.5);

% Create utility functions and constants
PI_2 = 2 * pi;
E_CONST = exp(1);

% Display initialization message
fprintf('   --> Variables initialized:\n');
fprintf('       global_param = %d\n', global_param);
fprintf('       PI_2 = %.6f\n', PI_2);
fprintf('       E_CONST = %.6f\n', E_CONST);

% Define helper function for later use
helper_ready = true;
fprintf('   --> Helper functions loaded and ready\n');

% Demonstrate scope
demo_variable = 'Defined in script_zero';
fprintf('   --> demo_variable: %s\n', demo_variable);

fprintf('   --> script_zero.m execution completed\n');