# Octave Plotting Cheatsheet

## Basic Plots

### Line Plots

```octave
x = 1:10;
y = x.^2;
plot(x, y)                    % Basic line plot
plot(x, y, 'r-', 'LineWidth', 2)  % Red line, width 2
plot(x, y, 'o-')             % Line with markers
semilogx(x, y)               % Log scale on x-axis
semilogy(x, y)               % Log scale on y-axis
loglog(x, y)                 % Log scale on both axes
```

### Multiple Plots

```octave
plot(x, y1, x, y2)           % Multiple series
plot(x, y1, 'r-', x, y2, 'b--')  % Different styles
hold on; plot(x, y2); hold off   % Add to existing plot
```

### Scatter Plots

```octave
scatter(x, y)                % Basic scatter
scatter(x, y, 50, 'red')     % Size 50, red color
scatter(x, y, size_array, color_array)  % Variable size/color
```

### Bar Charts

```octave
bar(x, y)                    % Vertical bars
barh(x, y)                   % Horizontal bars
bar(x, [y1, y2], 'grouped')  % Grouped bars
bar(x, [y1, y2], 'stacked')  % Stacked bars
```

### Histograms

```octave
histogram(data)              % Default bins
histogram(data, 20)          % 20 bins
histogram(data, 'BinWidth', 0.5)  % Bin width
hist(data, bins)             % Alternative syntax
```

## Statistical Plots

### Box Plots

```octave
boxplot(data)                % Single box plot
boxplot(data, groups)        % Grouped box plots
boxplot([data1(:), data2(:)], [ones(size(data1(:))), 2*ones(size(data2(:)))])
```

### Error Bars

```octave
errorbar(x, y, err)          % Symmetric error bars
errorbar(x, y, lower_err, upper_err)  % Asymmetric
```

## Subplots

### Creating Subplots

```octave
subplot(2, 2, 1);            % 2x2 grid, position 1
plot(x, y1);

subplot(2, 2, 2);            % Position 2
plot(x, y2);

% Alternative syntax
subplot(221); plot(x, y1);   % Same as subplot(2,2,1)
```

## 3D Plotting

### 3D Line/Scatter

```octave
plot3(x, y, z)               % 3D line plot
scatter3(x, y, z)            % 3D scatter plot
scatter3(x, y, z, size, color)  % With size/color
```

### Surface Plots

```octave
[X, Y] = meshgrid(-2:0.1:2, -2:0.1:2);
Z = X.^2 + Y.^2;
surf(X, Y, Z)                % Surface plot
mesh(X, Y, Z)                % Mesh plot
contour(X, Y, Z)             % Contour plot
contour3(X, Y, Z)            % 3D contour
```

### Surface Customization

```octave
surf(X, Y, Z)
shading interp               % Smooth shading
lighting gouraud             % Lighting model
camlight                     % Add light source
colormap(jet)                % Color scheme
colorbar                     % Color scale bar
```

## Plot Customization

### Titles and Labels

```octave
title('Plot Title')          % Main title
xlabel('X Axis Label')       % X-axis label
ylabel('Y Axis Label')       % Y-axis label
zlabel('Z Axis Label')       % Z-axis label (3D)
legend('Data 1', 'Data 2')   % Legend
text(x, y, 'Text')          % Add text annotation
```

### Axis Control

```octave
xlim([0, 10])               % Set x-axis limits
ylim([-5, 5])               % Set y-axis limits
zlim([0, 20])               % Set z-axis limits
axis equal                   % Equal scaling
axis tight                   % Tight axis limits
axis off                     % Hide axes
grid on                      % Show grid
grid off                     % Hide grid
```

### Line Styles and Colors

```octave
% Line styles
'-'     % Solid line
'--'    % Dashed line
':'     % Dotted line
'-.'    % Dash-dot line

% Colors
'r'     % Red
'g'     % Green
'b'     % Blue
'c'     % Cyan
'm'     % Magenta
'y'     % Yellow
'k'     % Black
'w'     % White

% Markers
'o'     % Circle
'*'     % Asterisk
'.'     % Point
'+'     % Plus
's'     % Square
'd'     % Diamond
'^'     % Triangle up
'v'     % Triangle down
```

### Advanced Styling

```octave
plot(x, y, 'Color', [0.5, 0.2, 0.8])  % RGB color
plot(x, y, 'LineWidth', 3)             % Line width
plot(x, y, 'MarkerSize', 10)           % Marker size
plot(x, y, 'MarkerFaceColor', 'red')   % Fill marker
```

## Figure Management

### Figure Control

```octave
figure                       % New figure window
figure(1)                    % Specific figure number
clf                          % Clear current figure
close                        % Close current figure
close all                    % Close all figures
```

### Figure Properties

```octave
set(gcf, 'Position', [100, 100, 800, 600])  % Set figure size/position
set(gca, 'FontSize', 12)     % Set axis font size
set(gca, 'LineWidth', 2)     % Set axis line width
```

## Specialized Plots

### Image Display

```octave
imagesc(matrix)              % Display matrix as image
imshow(image)                % Display image
colormap(gray)               % Grayscale colormap
colormap(jet)                % Jet colormap
colorbar                     % Add color bar
```

### Polar Plots

```octave
theta = 0:0.1:2*pi;
r = sin(theta);
polar(theta, r)              % Polar coordinate plot
```

### Stem Plots

```octave
stem(x, y)                   % Discrete data plot
stairs(x, y)                 % Step plot
```

## Colormaps

### Built-in Colormaps

```octave
colormap(gray)               % Grayscale
colormap(jet)                % Blue to red
colormap(hot)                % Black-red-yellow-white
colormap(cool)               % Cyan-magenta
colormap(spring)             % Magenta-yellow
colormap(summer)             % Green-yellow
colormap(autumn)             % Red-yellow
colormap(winter)             % Blue-green
```

### Custom Colormaps

```octave
% Create custom colormap
custom_map = [linspace(0,1,64)', zeros(64,1), linspace(1,0,64)'];
colormap(custom_map);
```

## Saving Plots

### Print to File

```octave
print('filename.png', '-dpng', '-r300')     % PNG at 300 DPI
print('filename.pdf', '-dpdf')              % PDF format
print('filename.eps', '-deps')              % EPS format
print('filename.jpg', '-djpeg', '-r150')    % JPEG format
```

### Save Figure

```octave
saveas(gcf, 'filename.fig')  % Save as Octave figure
saveas(gcf, 'filename.png')  % Save as PNG
```

## Animation

### Simple Animation

```octave
for i = 1:100
    plot(x, sin(x + i*0.1));
    axis([0, 10, -1, 1]);
    drawnow;                 % Update display
    pause(0.05);             % Brief pause
end
```

### 3D Rotation

```octave
for angle = 0:5:360
    view(angle, 30);
    drawnow;
    pause(0.1);
end
```

## Common Plot Types Quick Reference

| Plot Type | Function      | Example                   |
| --------- | ------------- | ------------------------- |
| Line plot | `plot()`      | `plot(x, y)`              |
| Scatter   | `scatter()`   | `scatter(x, y)`           |
| Bar chart | `bar()`       | `bar(categories, values)` |
| Histogram | `histogram()` | `histogram(data, 20)`     |
| Box plot  | `boxplot()`   | `boxplot(data, groups)`   |
| Surface   | `surf()`      | `surf(X, Y, Z)`           |
| Contour   | `contour()`   | `contour(X, Y, Z)`        |
| Image     | `imagesc()`   | `imagesc(matrix)`         |
| Polar     | `polar()`     | `polar(theta, r)`         |
| Stem      | `stem()`      | `stem(x, y)`              |

## Troubleshooting

### Common Issues

```octave
% Plot not showing
drawnow                      % Force plot update
figure                       % Create new figure window

% Axis problems
axis auto                    % Auto-scale axes
axis tight                   % Fit data tightly

% Legend issues
legend('off')                % Turn off legend
legend('Location', 'best')   % Auto-position legend
```

### Graphics Toolkit

```octave
graphics_toolkit('qt')       % Use Qt backend
graphics_toolkit('gnuplot')  % Use gnuplot backend
available_graphics_toolkits  % List available toolkits
```
