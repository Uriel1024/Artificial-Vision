% PROGRAMA QUE DADO UN CLASIFICADOR CON 8 CLASES TOMA UNA DECISIÓN
% TOMANDO COMO CRITERIO DE DECISIÓN A LA DISTANCIA EUCLIDEANA

clc
% DEFINIENDO LAS CLASES
clases = cell(1,8);

for i = 1:8
    inf  = ((i -1)*3 + 1);
    sup  = i*3;
    clases{i} = randi([inf,sup],2,5);
    
end

c1= randi([1,3],2,5);
c2= randi([3,6],2,5);
c3= randi([6,9],2,5);
c4= randi([9,12],2,5);
c5= randi([12,15],2,5);
c6= randi([15,18],2,5);
c7= randi([18,21],2,5);
c8= randi([21,24],2,5);


op = 121;

%ciclo principal para ejecutar iteradamente el problema hasta que el
%usuario decida terminarlo
while (op ~= -1)

vx=input('dame el valor de la coord en x=')
vy=input('dame el valor de la coord en y=')
vector=[vx;vy];


%CALCULANDO LOS PARÁMETROS DE CADA CLASE

media1=mean(c1,2);
media2=mean(C2,2);
media3=mean(c3,2);
media4=mean(c4,2);
media5=mean(c5,2);
media6=mean(c6,2);
media7=mean(c7,2);
media8=mean(c8,2);

%calculando su centroide:
dist1=norm(media1-vector);
dist2=norm(media2-vector);
dist3=norm(media3-vector);
dist4=norm(media4-vector);
dist5=norm(media5-vector);
dist6=norm(media6-vector);
dist7=norm(media7-vector);
dist8=norm(media8-vector);

dist_total=[dist1 dist2 dist3 dist4 dist5 dist6 dist7 dist8]

minimo=min(min(dist_total))
if minimo > 40
    fprintf("El vector no pertenece a niguna clase.\n\n")
else
dato=find(minimo==dist_total)


fprintf('el vector desconocido pertenece a la clase %d\n',dato)

% GRAFICANDO LAS CLASES
plot(c1(1,:),c1(2,:),'ro','MarkerSize',10,'MarkerFaceColor','r')
grid on
hold on
plot(C2(1,:),C2(2,:),'bo','MarkerSize',10,'MarkerFaceColor','b')
plot(c3(1,:),c3(2,:),'go','MarkerSize',10,'MarkerFaceColor','k')
plot(c4(1,:),c3(2,:),'bo','MarkerSize',10,'MarkerFaceColor','m')
plot(c5(1,:),c3(2,:),'go','MarkerSize',10,'MarkerFaceColor','c')
plot(c6(1,:),c3(2,:),'bo','MarkerSize',10,'MarkerFaceColor','y')
plot(c7(1,:),c3(2,:),'go','MarkerSize',10,'MarkerFaceColor','w')
plot(c3(1,:),c3(2,:),'bo','MarkerSize',10,'MarkerFaceColor','b')


plot(vector(1,:),vector(2,:),'go','MarkerSize',10,'MarkerFaceColor','g')

legend('clase1','clase2','clase3','clase4','clase5','clase6','clase7','clase8',' vector')
end
op = input('Deseas seguir calculando las distancias del vector (-1 para terminar el programa):')
end
disp('fin de proceso....')