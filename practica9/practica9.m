%practica 9 
clc
clear
close all 

clases = [
        % Clase 1 
        % x,y,z, clase a la q pertenece (0/1)
        0, 0, 0, 0;  
        1, 0, 0, 0;   
        0, 1, 0, 0;   
        0, 0, 1, 0;   

        % Clase 2 
        1, 1, 0, 1;   
        1, 0, 1, 1;  
        0, 1, 1, 1;  
        1, 1, 1, 1;  
    ];

b= input("Ingresa el valro del bias: ");
z = input("Ingresa los valores iniciales en z: ");
y = input("Ingresa los valores iniciales en y: ");
x = input("Ingresa los valores iniciales en x: ");


%vector inicial 
vector = [b,z,y,x];


contador = 0;
converge = false;

fprintf("inicio del entrenamiento del perceptron. ")

while ~converge
    converge = true;
    contador = contador + 1;
    %asumimos que el plano original encaja
    for j = 1:size(clases,1)
        fsal = [1,clases(j,1:3)];
        y = clases(j,4); 

        x = vector  * fsal';
        if y == 0
            if x >= 0
                vector = vector - fsal;
                converge = false;
            end
        else
            if x<=0
                vector = vector + fsal;
                converge = false;
            end
        end
    end


    %para ver cuantas iteraciones tardo en encontrar el plano optimo
    fprintf("%d iteracion con el sig vector:", contador)
    disp(vector)
    
    if converge
        break;
    end

end


fprintf("\n\nLa neurona alcanzo el plano optimo despues de %d  iteraciones con el vector", contador);
disp(vector)


% vector = [b, w_x, w_y, w_z]
b = vector(1);
wx = vector(2); % Corresponde a la 1ra característica (columna 1 de clases)
wy = vector(3); % Corresponde a la 2da característica (columna 2 de clases)
wz = vector(4); % Corresponde a la 3ra característica (columna 3 de clases)


% --- 5. Graficado de Clases y Plano Separador ---

% Crear figura 3D
figure;
ax = axes('Parent', gcf, 'NextPlot', 'add');
view(ax, 3);
grid on;
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$y$', 'Interpreter', 'latex');
zlabel('$z$', 'Interpreter', 'latex');
title(sprintf('Clases y Plano Separador del Perceptrón: %.2f + %.2fx + %.2fy + %.2fz = 0', b, wx, wy, wz));
axis equal;
xlim([-0.2, 1.2]);
ylim([-0.2, 1.2]);
zlim([-0.2, 1.2]);

% 5.1. Plotear los puntos de las clases
clase_0_data = clases(clases(:,4) == 0, 1:3);
clase_1_data = clases(clases(:,4) == 1, 1:3);

scatter3(clase_0_data(:,1), clase_0_data(:,2), clase_0_data(:,3), ...
    100, 'b', 'o', 'filled', 'DisplayName', 'Clase 0 (Target 0)');
scatter3(clase_1_data(:,1), clase_1_data(:,2), clase_1_data(:,3), ...
    100, 'r', '^', 'filled', 'DisplayName', 'Clase 1 (Target 1)');

% 5.2. Plotear el plano separador (b + wx*x + wy*y + wz*z = 0)
% Creamos una malla de puntos x e y
[x_plane, y_plane] = meshgrid(-0.5:0.1:1.5, -0.5:0.1:1.5);

% Calculamos z a partir de la ecuación del plano: z = (-b - wx*x - wy*y) / wz
if abs(wz) > 1e-6 % Aseguramos que el peso de Z no es cero (plano no vertical)
    z_plane = (-b - wx*x_plane - wy*y_plane) / wz;
    
    % Dibujar la superficie
    surf(x_plane, y_plane, z_plane, 'FaceColor', 'g', 'FaceAlpha', 0.5, ...
         'EdgeColor', 'none', 'DisplayName', 'Plano Separador');
else
    % Manejo de caso donde wz es cero (plano perpendicular al eje z)
    fprintf("Advertencia: wz es cero. El plano es perpendicular al eje XY, no se gráfica la superficie fácilmente.\n");
end

% Añadir Leyenda
legend('Location', 'best');



fprintf("\n\nFinal del programa, ahi nos voidmos");


