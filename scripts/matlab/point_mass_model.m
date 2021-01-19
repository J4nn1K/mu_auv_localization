% Symbolic Computation of simple point mass model for EKF
% Still work in progress
% Author: Nathalie Bauschmann

close all; clear all; clc;

%% Define Variables
syms x y z R P Y real
syms dx dy dz rr pr yr real
syms dt

% pose
position = [x; y; z];
orientation = [R; P; Y]; % euler angles, rotating around static axes, in order: x-y-z

% linear and angular velocities in body frame
lin_vel = [dx; dy; dz];
ang_vel = [rr; pr; yr];

% state vector
s = [position; orientation; lin_vel; ang_vel];

%%
% rotation matrix from euler angles to transform from body frame -> map
% frame

% roll -> rotation around x-axis
rot_roll = [1, 0, 0;
            0, cos(R), -sin(R);
            0, sin(R), cos(R)];

% pitch -> rotation around y-axis
rot_pitch = [cos(P), 0, sin(P);
             0, 1, 0;
             -sin(P), 0, cos(P)];

% yaw -> rotation around z-axis
rot_yaw = [cos(Y), -sin(Y), 0;
           sin(Y), cos(Y), 0;
           0, 0, 1];

rot_map_body = simplify(rot_yaw * rot_pitch * rot_roll);
rot_body_map = simplify(inv(rot_map_body));

%%
% compute velocity in map frame
lin_vel_map = rot_map_body * lin_vel;
% new position in map frame
new_position = position + dt*lin_vel_map;
% new orientation in map frame
new_orientation = orientation;   % todo

% Don't change velocities
% new linear velocities in body frame
new_lin_vel = lin_vel;
% new angular velocities in body frame
new_ang_vel = ang_vel; 

f = simplify([new_position; new_orientation; new_lin_vel; new_ang_vel])
A = simplify(jacobian(f, s))

