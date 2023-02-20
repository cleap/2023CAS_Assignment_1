%Logistic Equation: x(n+1) = r * x(n) * (1 - x(n))

% Set the parameters for the simulation
r_vals = [2, 3.9]; % growth rate
x0_vals = [0.5, 0.50000001]; % initial population value
t_end = 100; % number of time steps

% Simulate the logistic map
x = zeros(1, t_end);

for ri = 1:(length(r_vals))
    r = r_vals(ri);

    for xi = 1:(length(x0_vals))
        x0 = x0_vals(xi);
        x(1) = x0;

        for ni = 2:t_end
            x(ni) = r * x(ni-1) * (1 - x(ni-1));
        end

        % Plot the results
        if r == 2
            figure(1)
            hold on
            plot(1:t_end, x, '-', 'LineWidth', 1.5);
            xlabel('Time');
            ylabel('Population');
            legend(['x0 = ', num2str(x0_vals(1))], ['x0 = ',...
                num2str(x0_vals(2)), '0000001']);
            title(['Logistic Map | r = ', num2str(r)]);
        else
            figure(2)
            hold on
            plot(1:t_end, x, '-', 'LineWidth', 1.5);
            xlabel('Time');
            ylabel('Population');
            legend(['x0 = ', num2str(x0_vals(1))], ['x0 = ',...
                num2str(x0_vals(2)), '0000001']);
            title(['Logistic Map | r = ', num2str(r)]);
        end
    end
end