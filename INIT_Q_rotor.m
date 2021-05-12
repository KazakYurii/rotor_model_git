%Parametrs
Ki = 1e-1;%Kof Integrator
Kr = 1000;%Reward Kof
Kd = -800;%isdone Kof
Ks = 10;%signal Kof
%Traning options
Umotion = 5000;
MaxEp = 5000; %max epochs
ValueStop = 35; %the stop value
%
ustavka=3.2e-3;%уставочное значение для Х3
%Допустительный онтервал
dumax=ustavka+0.2e-3;
dumin=ustavka-0.2e-3;
p0min=1.05e5;
p0max=1.1e5;
p1min=1e5;
p1max=1.08e5;

%Model
mdl = 'Q_learning_rotor_v3_min_M_imp_not_disb';
open_system(mdl)
nObs=3;%number of observations
nAct=1;%number of actions
action = 0.6e3;%parametr actInfo

%Train the agent using the train function. This is a computationally intensive process that takes several minutes to complete. 
%To save time while running this example, load a pretrained agent by setting doTraining to false. 
%To train the agent yourself, set doTraining to true.
doTraining = true;

%Assign the agent block path information, and create rlNumericSpec and rlFiniteSetSpec objects for the observation and action information. 
%You can use dot notation to assign property values of the rlNumericSpec and rlFiniteSetSpec objects.
agentBlk = [mdl '/RL Agent'];
obsInfo = rlNumericSpec([nObs 1])
%actInfo = rlNumericSpec([nAct 1])%,'LowerLimit',-10*ones(2,1),...
                              %'UpperLimit',10*ones(2,1))

%actInfo = rlFiniteSetSpec([1e5,1.01e5,1.02e5,1.03e5,1.04e5,1.05e5,1.06e5,1.07e5,1.08e5])                              
%actInfo = rlFiniteSetSpec({[-action;-action],[action;-action],[-action;action],[action;action],[0;-action],...
%                            [-action;0],[0;action],[action;0],[0;0]})
actInfo = rlFiniteSetSpec([-action;0;action;-2*action;2*action])
%                            [-action;0],[0;action],[action;0],[0;0]})
%actInfo = rlFiniteSetSpec([1,2,3])
% %actInfo = rlFiniteSetSpec({[-0.1; -0.1], [0 ;-0.1], [0.1; -0.1], ...
%                             [-0.1; 0], [0; 0], [0.1 ;0], ...
%                             [0.1 ;0.1], [0; 0.1], [0.1; 0.1]})
%actInfo = rlFiniteSetSpec([-1 0 -1])                        
obsInfo.Name = 'observations';
actInfo.Name = 'voltage';

%Create the reinforcement learning environment for the Simulink model using information extracted in the previous steps.
env = rlSimulinkEnv(mdl,agentBlk,obsInfo,actInfo)

% % load predefined environment
% env = rlPredefinedEnv("CartPole-Discrete");
% 
% % get observation and specification info
% obsInfo = getObservationInfo(env);
% actInfo = getActionInfo(env);

%You can also include a reset function using dot notation. For this example, consider randomly initializing theta0 in the model workspace.
%env.ResetFcn = @(in) setVariable(in,'theta0',randn,'Workspace',mdl)
%Specify the simulation time Tf and the agent sample time Ts in seconds
Tss = 0.1;
Tf = 4;

%Create DQN agent
%A DQN agent approximates the long-term reward given observations and actions using a critic value function representation. 
%To create the critic, first create a deep neural network with two inputs, the state and action, and two outputs. 
%The input size of the state path is [2 1 1] since the environment provides 2 observations. 
%For more information on creating a deep neural network value function representation, see Create Policy and Value Function Representations.
statePath = [
    imageInputLayer([nObs 1 1],'Normalization','none','Name','state')
    fullyConnectedLayer(14,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(18,'Name','CriticStateFC2')]
actionPath = [
    imageInputLayer([nAct 1 1],'Normalization','none','Name','action')
    fullyConnectedLayer(18,'Name','CriticActionFC1','BiasLearnRateFactor',0)]
commonPath = [
    additionLayer(2,'Name','add')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(1,'Name','output')]
criticNetwork = layerGraph();
criticNetwork = addLayers(criticNetwork,statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');
%View the critic network configuration.
figure
plot(criticNetwork)

%Specify options for the critic representation using rlRepresentationOptions.
criticOptions = rlRepresentationOptions('LearnRate',0.001,'GradientThreshold',1,'Optimizer',"adam");%);
%criticOptions.OptimizerParameters.Momentum = 0.8;


%Create the critic representation using the specified deep neural network and options. 
%You must also specify the action and observation info for the critic, which you obtain from the environment interface. 
%For more information, see rlQValueRepresentation.
obsInfo = getObservationInfo(env)
actInfo = getActionInfo(env)
critic = rlQValueRepresentation(criticNetwork,obsInfo,actInfo,...
    'Observation',{'state'},'Action',{'action'},criticOptions)

%To create the DQN agent, first specify the DQN agent options using rlDQNAgentOptions.
agentOptions = rlDQNAgentOptions(...
    'SampleTime',Tss,...
    'TargetSmoothFactor',1e-3,...
    'SaveExperienceBufferWithAgent',1,...
    'ExperienceBufferLength',100000,... 
    'UseDoubleDQN',false,...
    'DiscountFactor',0.85,...
    'MiniBatchSize',250);

%Then, create the DQN agent using the specified critic representation and agent options. For more information, see rlDQNAgent.
agent = rlDQNAgent(critic,agentOptions)

%Training options
trainingOptions = rlTrainingOptions( ...
    'MaxEpisodes',MaxEp,...
    'MaxStepsPerEpisode',(Tf/Tss),...
    'ScoreAveragingWindowLength',5,...
    'Verbose',false,...
    'Plots','training-progress', ...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',ValueStop,...
    'SaveAgentCriteria','EpisodeReward',...
    'SaveAgentValue',ValueStop);



if doTraining
    % Train the agent.
    trainingStats = train(agent,env,trainingOptions);
    
    curDir = pwd;
saveDir = 'savedAgents';
cd(saveDir)
save(['trainedAgent_2D_' datestr(now,'mm_DD_YYYY_HHMM')],'agent','trainingStats');
cd(curDir)
    
else
    % Load pretrained agent for the example.
    load('SimulinkPendulumDQN.mat','agent');
end

