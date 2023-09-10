clc;
clear all;
close all;
tic;
%% DATA
data=load('acc_gyro.txt');%load the data
t=data(:,1);              %time
xAcc=data(:,2);           %acceleration along x-axis
yAcc=data(:,3);           %acceleration along y-axis
zAcc=data(:,4);           %acceleration along z-axis
p=data(:,5);              %pitch
r=data(:,6);              %roll
y=data(:,7);              %yaw

data(:,2:4)=9.81*data(:,2:4);

%plot acceleration along the x-axis
figure(1)
plot(t,xAcc,'.r');
grid on
xlabel('Time (s)');
ylabel('Acceleration along the x-axis (G)');

%plot acceleration along the y-axis
figure(2)
plot(t,yAcc,'.b');
grid on
ylim([-1.2 0.4]);
xlabel('Time (s)');
ylabel('Acceleration along the y-axis (G)');

%plot acceleration along the z-axis
figure(3)
plot(t,zAcc,'.g');
grid on
ylim([-1.4 0.4]);
xlabel('Time (s)');
ylabel('Acceleration along the z-axis (G)');

%plot all accelerations together
figure(4)
plot(t,xAcc,'r');
grid on
hold on
plot(t,yAcc,'b');
plot(t,zAcc,'g');
ylim([-1.4 1.4]);
xlabel('Time (s)');
ylabel('Acceleration (G)');
legend('x-direction','y-direction','z-direction','northeast');

%plot pitch
figure(5)
plot(t,p,'.r');
grid on
ylim([-2.5 2.5]);
xlabel('Time (s)');
ylabel('Pitch (rad/s)');

%plot roll
figure(6)
plot(t,r,'.b');
grid on
ylim([-3 3]);
xlabel('Time (s)');
ylabel('Roll (rad/s)');

%plot yaw
figure(7)
plot(t,y,'.g');
grid on
ylim([-2.5 2.5]);
xlabel('Time (s)');
ylabel('Yaw (rad/s)');

%plot all angles together
figure(8)
plot(t,p,'r');
grid on
hold on
plot(t,r);
plot(t,y,'g');
ylim([-3 3]);
xlabel('Time (s)');
ylabel('Angle (rad/s)');
legend('pitch','roll','yaw','northeast');
%%
velInitial=0;
sampTime=0.05;
motion=0;
motion1=0;
for i=1061:1:1199
    velFinal=velInitial+((data(i,2)/9.81)*sampTime);
    velAvg=(velFinal+velInitial)/2;
    motion=motion+velAvg*sampTime;
    motion1=motion1+((velInitial*sampTime)+(0.5*data(i,2)*(sampTime^2)));
    velInitial=velFinal;
end
%%
% clear data
toc;