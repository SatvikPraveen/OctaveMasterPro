# GNU Octave Syntax Cheatsheet

## Variables & Data Types

### Basic Assignment

```octave
x = 5;                    % Scalar
y = [1, 2, 3];           % Row vector
z = [1; 2; 3];           % Column vector
A = [1, 2; 3, 4];        % Matrix
str = 'hello';           % String
cell_array = {1, 'text', [1,2]};  % Cell array
```

### Data Type Checking

```octave
class(x)                 % Get data type
ischar(str)              % Check if character
isnumeric(x)             % Check if numeric
ismatrix(A)              % Check if matrix
size(A)                  % Get dimensions
length(y)                % Get length
numel(A)                 % Total number of elements
```

## Array Creation & Manipulation

### Array Creation

```octave
zeros(3, 4)              % 3x4 matrix of zeros
ones(2, 5)               % 2x5 matrix of ones
eye(3)                   % 3x3 identity matrix
rand(2, 3)               % Random values [0,1]
randn(2, 3)              % Normal random values
linspace(0, 10, 5)       % 5 points from 0 to 10
1:5                      % [1, 2, 3, 4, 5]
1:2:10                   % [1, 3, 5, 7, 9]
```

### Array Manipulation

```octave
A(2, 3)                  % Element at row 2, col 3
A(2, :)                  % Entire row 2
A(:, 3)                  % Entire column 3
A(1:2, 2:3)              % Submatrix
A'                       % Transpose
A(:)                     % Column vector (all elements)
reshape(A, 4, 2)         % Reshape to 4x2
```

## Operators

### Arithmetic Operators

```octave
A + B                    % Matrix addition
A - B                    % Matrix subtraction
A * B                    % Matrix multiplication
A .* B                   % Element-wise multiplication
A / B                    % Matrix division (A * B^-1)
A ./ B                   % Element-wise division
A ^ 2                    % Matrix power
A .^ 2                   % Element-wise power
```

### Comparison Operators

```octave
A == B                   % Element-wise equality
A ~= B                   % Element-wise inequality
A < B                    % Less than
A <= B                   % Less than or equal
A > B                    % Greater than
A >= B                   % Greater than or equal
```

### Logical Operators

```octave
A & B                    % Element-wise AND
A | B                    % Element-wise OR
~A                       % NOT
any(A)                   % True if any element is true
all(A)                   % True if all elements are true
```

## Control Structures

### If-Else Statements

```octave
if condition
    % code
elseif other_condition
    % code
else
    % code
end
```

### Loops

```octave
% For loop
for i = 1:n
    % code
end

% While loop
while condition
    % code
end

% Break and continue
for i = 1:10
    if i == 5
        continue;        % Skip iteration
    end
    if i == 8
        break;           % Exit loop
    end
end
```

### Switch Statement

```octave
switch variable
    case value1
        % code
    case {value2, value3}
        % code for multiple values
    otherwise
        % default case
end
```

## Functions

### Function Definition

```octave
function [output1, output2] = myfunction(input1, input2)
    % Function documentation
    output1 = input1 * 2;
    output2 = input2 + 1;
end
```

### Anonymous Functions

```octave
f = @(x) x^2 + 2*x + 1;  % Define function
result = f(5);           % Call function
```

### Variable Arguments

```octave
function result = myfunc(varargin)
    n_args = nargin;     % Number of input arguments
    if n_args > 0
        result = varargin{1};
    end
end
```

## File Operations

### Script Files

```octave
script_name              % Run script file
run('script_name.m')     % Alternative way
```

### Data I/O

```octave
% Save/Load variables
save('data.mat', 'var1', 'var2');
load('data.mat');

% Text files
data = load('file.txt');     % Load numeric data
save('file.txt', 'data', '-ascii');

% CSV files
data = readtable('file.csv');
writetable(data, 'output.csv');
```

## Built-in Functions

### Mathematical Functions

```octave
sin(x), cos(x), tan(x)   % Trigonometric
exp(x), log(x), log10(x) % Exponential/logarithmic
sqrt(x), abs(x)          % Square root, absolute value
round(x), floor(x), ceil(x)  % Rounding
min(x), max(x)           % Min/max values
sum(x), mean(x), std(x)  % Statistics
```

### Matrix Functions

```octave
inv(A)                   % Matrix inverse
det(A)                   % Determinant
rank(A)                  % Matrix rank
trace(A)                 % Trace (sum of diagonal)
eig(A)                   % Eigenvalues
svd(A)                   % Singular value decomposition
```

### Array Functions

```octave
find(condition)          % Find indices where condition is true
sort(x)                  % Sort array
unique(x)                % Unique elements
diff(x)                  % Differences between elements
cumsum(x)                % Cumulative sum
```

## String Operations

### String Manipulation

```octave
str = 'Hello World';
length(str)              % String length
str(1:5)                 % Substring
upper(str)               % Convert to uppercase
lower(str)               % Convert to lowercase
strcmp(str1, str2)       % String comparison
strcat(str1, str2)       % String concatenation
sprintf('Value: %d', x)  % Formatted string
```

## Plotting Basics

### Basic Plots

```octave
plot(x, y)               % Line plot
scatter(x, y)            % Scatter plot
bar(x, y)                % Bar plot
histogram(data)          % Histogram
```

### Plot Customization

```octave
title('My Plot')         % Add title
xlabel('X axis')         % X-axis label
ylabel('Y axis')         % Y-axis label
legend('Data 1', 'Data 2')  % Add legend
grid on                  % Show grid
hold on                  % Keep current plot
hold off                 % Release plot hold
```

## Error Handling

### Try-Catch

```octave
try
    % risky code
    result = risky_operation();
catch ME
    % error handling
    fprintf('Error: %s\n', ME.message);
    result = default_value;
end
```

### Assertions

```octave
assert(condition, 'Error message');
assert(x > 0, 'x must be positive');
```

## Package Management

### Package Operations

```octave
pkg list                 % List installed packages
pkg load statistics      % Load package
pkg install -forge signal  % Install package from Octave Forge
```

## Workspace Management

### Variable Management

```octave
who                      % List variables
whos                     % List variables with details
clear x y                % Clear specific variables
clear all                % Clear all variables
clc                      % Clear command window
```

### Path Management

```octave
pwd                      % Current directory
cd('path')               % Change directory
addpath('path')          % Add to search path
rmpath('path')           % Remove from search path
path                     % Show current path
```

## Debugging

### Debugging Commands

```octave
dbstop in file at line   % Set breakpoint
dbcont                   % Continue execution
dbstep                   % Step to next line
dbstack                  % Show call stack
dbquit                   % Exit debugger
```

### Performance

```octave
tic; code; toc           % Time execution
profile on; code; profile off; profshow  % Profile code
```

## Common Patterns

### Vectorization

```octave
% Avoid loops when possible
% Bad:
for i = 1:length(x)
    y(i) = x(i)^2;
end

% Good:
y = x .^ 2;
```

### Logical Indexing

```octave
% Find elements meeting condition
mask = x > 5;            % Logical mask
result = x(mask);        % Elements where x > 5
x(x < 0) = 0;           % Set negative values to zero
```

### Cell Array Operations

```octave
cellfun(@func, cell_array)     % Apply function to each cell
cell2mat(cell_array)           % Convert to matrix
mat2cell(matrix, sizes)        % Convert matrix to cells
```

## Quick Reference

| Operation           | Syntax          | Example               |
| ------------------- | --------------- | --------------------- |
| Comment             | `%`             | `% This is a comment` |
| Line continuation   | `...`           | `x = 1 + 2 + ...`     |
| Suppress output     | `;`             | `x = 5;`              |
| Multiple statements | `,` or `;`      | `x = 1; y = 2;`       |
| Help                | `help function` | `help plot`           |
| Documentation       | `doc function`  | `doc fft`             |

## Common Gotchas

1. **Array vs Matrix Multiplication**: Use `.*` for element-wise, `*` for matrix multiplication
2. **Indexing**: Octave uses 1-based indexing (not 0-based)
3. **Semicolons**: Suppress output with `;`
4. **String Comparison**: Use `strcmp()`, not `==`
5. **Function Handles**: Use `@` prefix: `@sin`, `@(x) x^2`

## Quick Help Commands

```octave
help                     % General help
help plot                % Help for specific function
doc plot                 % Documentation
lookfor keyword          % Search for functions
which function           % Find function location
```
