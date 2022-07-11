import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.ndimage import uniform_filter, gaussian_filter

from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import lib.draw as draw

def plot_pitch_control(frame, grid, control, modelname=None, title=None, dpi=120, subplot=None, savefig=None):
    '''
    Description
    -------
    Plots the pitch control visualization using the scatter function

    Parameters
    -------
        frame: a dataframe containing the positions x, y of all players and the ball, and both bgcolor and edgecolor
        grid: a dataframe containing the coordinates x, y used to generate the pitch control vector
        control: a series containing the values of control for each coordinate in the grid
        modelname: text to write in the figure
        dpi: adjust image dpi
        subplot: use to plot only in one subplot. None to ignore.
        savefig: use to give a directory to save the figure. None to ignore.

    Returns
    -------
        None
    '''

    # Add subplot if requested
    if subplot != None:
        plt.subplot(subplot)

    # Drawing background
    draw.pitch(dpi=dpi)

    # Plotting pitch control
    ## Old, deprecated way of plotting the grid
    ## plt.scatter(grid.x, grid.y, s=20, marker='s', c=control, cmap='seismic', alpha=0.2)
    ## plt.clim(control.min(), control.max())
    plt.imshow(np.rot90(control.to_numpy().reshape((106,69))), 
        extent=(0, 100, 0, 100), interpolation='bicubic',
        vmin=control.min(),vmax=control.max(),
        cmap='seismic',alpha=0.4, aspect=6.8/10.5, zorder=2)

    
    # Plotting the elements of both teams
    _players = frame[frame.player != 0]
    plt.scatter(_players.x, _players.y, s=100, c=_players.bgcolor.values, edgecolors=_players.edgecolor, zorder=10)
    _ball = frame[frame.player == 0]
    plt.scatter(_ball.x, _ball.y, s=50, c=_ball.bgcolor.values, edgecolors=_ball.edgecolor, zorder=11)


    # Writing name of the model on the figure
    if modelname != None:
        plt.text(1, 1, modelname, fontsize='xx-large',
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    
    # Writing a title
    if title != None:
        plt.text(1, 99, title, fontsize='xx-large', va='top',
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])


    # End clauses to:
    ## Save figure in a directory
    if savefig != None:
        plt.savefig(savefig, bbox_inches='tight')
        plt.clf()
    ## Wait for the remaining subplots
    elif subplot != None:
        return
    ## Plot the figure
    else:
        plt.show()

class KNNPitchControl:
    '''
    Description
    -------
    Class defining the KNN version of pitch control. Default parameters generate a Voronoi approximation.

    Init Parameters
    -------
        lags [list(int)]: list of values indicating how many frames we extrapolate current speed. Each value generates a model, all models' outputs are summed to obtain the final result.
        distance_basis: base value for decaing control with increased distance. None to ignore.
        smoothing: size of the smoothing window. None to ignore.
        team1_id: name of a team to define the label.
        k: number of neighbors for the KNN model.
        n_jobs: for multiprocessing purposes.
    
    Methods
    -------
        predict (xy -> dataframe containing the positions x, y of all players and the ball): returns an array with the pitch control values according to the predefined grid
    '''

    def __init__(self, lags=[0], distance_basis=None, smoothing=None, team1_id='attack', k=1, n_jobs=-1):
        self.model = KNeighborsClassifier(n_neighbors=k, n_jobs=n_jobs)

        self.smoothing = smoothing
        self.lags = lags
        self.distance_basis = distance_basis
        self.team1_id = team1_id

        self.grid = pd.DataFrame([[i/1.05, j/0.68] for i in range(106) for j in range(69)], columns=['x','y'])
        self.grid['model'] = 'KNNPitchControl'

    def predict(self, xy):
        # Initializing control variable as an array of 0s
        self.grid['control'] = 0

        # For each lag set by the user
        for lag in self.lags:
            # Copy player data (ignoring the ball)
            _xy = xy[xy.player!=0].copy()

            # Extrapolate position according to the lag defined
            _xy['x'] += lag * _xy.dx 
            _xy['y'] += lag * _xy.dy 

            # Calculate the label according to the team1_id defined in innitialization
            _xy['team'] = (xy.team == self.team1_id) * 2 - 1

            # Fitting the KNN model
            self.model.fit(_xy[['x','y']], _xy['team'])

            # Calculating distance_factor if parameter is enabled.
            if self.distance_basis != None:
                distances, _ = self.model.kneighbors(self.grid[['x', 'y']])
                distance_factor = self.distance_basis ** (1 - distances[:,0]/40)
            else:
                distance_factor = 1

            # Sum pitch control of the current lag to the total control
            self.grid['control'] += distance_factor * self.model.predict(self.grid[['x','y']])

        # Calculate smoothed variable if parameter is enabled
        if self.smoothing != None:
            for coord in self.grid.x.unique():
                # Across X axis
                self.grid.loc[self.grid.x == coord, 'control'] = self.grid.loc[self.grid.x == coord, 'control'
                    ].rolling(self.smoothing, min_periods=1, center=True).mean().rolling(self.smoothing, min_periods=1, center=True).mean()
                # Across Y axis
                self.grid.loc[self.grid.y == coord, 'control'] = self.grid.loc[self.grid.y == coord, 'control'
                    ].rolling(self.smoothing, min_periods=1, center=True).mean().rolling(self.smoothing, min_periods=1, center=True).mean()
            

        # Return the grid, normalized for the interval 0-1
        return self.grid['control'] / self.grid['control'].max()




# Laurie Shaw's implementation of Spearman's pitch control
class spearman_player(object):
    """
    Class defining a player object that stores position, velocity, time-to-intercept and pitch control contributions for a player
    
    __init__ Parameters
    -----------
    pid: id (jersey number) of player
    team: row of tracking data for team
    teamname: team name "Home" or "Away"
    params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
    
    methods include:
    -----------
    simple_time_to_intercept(r_final): time take for player to get to target position (r_final) given current position
    probability_intercept_ball(T): probability player will have controlled ball at time T given their expected time_to_intercept
    
    """
    # player object holds position, velocity, time-to-intercept and pitch control contributions for each player
    def __init__(self,pid,x,y,dx,dy,params):
        self.id = pid
        self.vmax = params['max_player_speed'] # player max speed in m/s. Could be individualised
        self.reaction_time = params['reaction_time'] # player reaction time in 's'. Could be individualised
        self.tti_sigma = params['tti_sigma'] # standard deviation of sigmoid function (see Eq 4 in Spearman, 2018)
        self.lambda_att = params['lambda_att'] # standard deviation of sigmoid function (see Eq 4 in Spearman, 2018)
        self.is_gk = False ## TODO: We do not distinguish between gks and the rest
        self.lambda_def = params['lambda_gk'] if self.is_gk else params['lambda_def'] # factor of 3 ensures that anything near the GK is likely to be claimed by the GK
        self.set_position_and_velocity(x, y, dx, dy)
        self.PPCF = 0. # initialise this for later
        
    def set_position_and_velocity(self, x, y, dx, dy):
        self.position = np.array( [ x, y ] )
        self.inframe = not np.any( np.isnan(self.position) )
        self.velocity = np.array( [ dx, dy ] )
        if np.any( np.isnan(self.velocity) ):
            self.velocity = np.array([0.,0.])
        
    def simple_time_to_intercept(self, r_final):
        self.PPCF = 0. # initialise this for later
        # Time to intercept assumes that the player continues moving at current velocity for 'reaction_time' seconds
        # and then runs at full speed to the target position.
        r_reaction = self.position + self.velocity*self.reaction_time
        self.time_to_intercept = self.reaction_time + np.linalg.norm(r_final-r_reaction)/self.vmax
        return self.time_to_intercept

    def probability_intercept_ball(self,T):
        # probability of a player arriving at target location at time 'T' given their expected time_to_intercept (time of arrival), as described in Spearman 2018
        f = 1/(1. + np.exp( -np.pi/np.sqrt(3.0)/self.tti_sigma * (T-self.time_to_intercept) ) )
        return f

def calculate_spearman_pitch_control_at_target(target_position, attacking_players, defending_players, ball_start_pos, params):
    """
    Calculates the pitch control probability for the attacking and defending teams at a specified target position on the ball.
    
    Parameters
    -----------
        target_position: size 2 numpy array containing the (x,y) position of the position on the field to evaluate pitch control
        attacking_players: list of 'player' objects (see player class above) for the players on the attacking team (team in possession)
        defending_players: list of 'player' objects (see player class above) for the players on the defending team
        ball_start_pos: Current position of the ball (start position for a pass). If set to NaN, function will assume that the ball is already at the target position.
        params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
        
    Returrns
    -----------
        PPCFatt: Pitch control probability for the attacking team
        PPCFdef: Pitch control probability for the defending team ( 1-PPCFatt-PPCFdef <  params['model_converge_tol'] )
    """
    # calculate ball travel time from start position to end position.
    if ball_start_pos is None or any(np.isnan(ball_start_pos)): # assume that ball is already at location
        ball_travel_time = 0.0 
    else:
        # ball travel time is distance to target position from current ball position divided assumed average ball speed
        ball_travel_time = np.linalg.norm( target_position - ball_start_pos )/params['average_ball_speed']
    
    # first get arrival time of 'nearest' attacking player (nearest also dependent on current velocity)
    tau_min_att = np.nanmin( [p.simple_time_to_intercept(target_position) for p in attacking_players] )
    tau_min_def = np.nanmin( [p.simple_time_to_intercept(target_position ) for p in defending_players] )
    
    # check whether we actually need to solve equation 3
    if tau_min_att-max(ball_travel_time,tau_min_def) >= params['time_to_control_def']:
        # if defending team can arrive significantly before attacking team, no need to solve pitch control model
        return 0., 1.
    elif tau_min_def-max(ball_travel_time,tau_min_att) >= params['time_to_control_att']:
        # if attacking team can arrive significantly before defending team, no need to solve pitch control model
        return 1., 0.
    else: 
        # solve pitch control model by integrating equation 3 in Spearman et al.
        # first remove any player that is far (in time) from the target location
        attacking_players = [p for p in attacking_players if p.time_to_intercept-tau_min_att < params['time_to_control_att'] ]
        defending_players = [p for p in defending_players if p.time_to_intercept-tau_min_def < params['time_to_control_def'] ]
        # set up integration arrays
        dT_array = np.arange(ball_travel_time-params['int_dt'],ball_travel_time+params['max_int_time'],params['int_dt']) 
        PPCFatt = np.zeros_like( dT_array )
        PPCFdef = np.zeros_like( dT_array )
        # integration equation 3 of Spearman 2018 until convergence or tolerance limit hit (see 'params')
        ptot = 0.0
        i = 1
        while 1-ptot>params['model_converge_tol'] and i<dT_array.size: 
            T = dT_array[i]
            for player in attacking_players:
                # calculate ball control probablity for 'player' in time interval T+dt
                dPPCFdT = (1-PPCFatt[i-1]-PPCFdef[i-1])*player.probability_intercept_ball( T ) * player.lambda_att
                # make sure it's greater than zero
                assert dPPCFdT>=0, 'Invalid attacking player probability (calculate_pitch_control_at_target)'
                player.PPCF += dPPCFdT*params['int_dt'] # total contribution from individual player
                PPCFatt[i] += player.PPCF # add to sum over players in the attacking team (remembering array element is zero at the start of each integration iteration)
            for player in defending_players:
                # calculate ball control probablity for 'player' in time interval T+dt
                dPPCFdT = (1-PPCFatt[i-1]-PPCFdef[i-1])*player.probability_intercept_ball( T ) * player.lambda_def
                # make sure it's greater than zero
                assert dPPCFdT>=0, 'Invalid defending player probability (calculate_pitch_control_at_target)'
                player.PPCF += dPPCFdT*params['int_dt'] # total contribution from individual player
                PPCFdef[i] += player.PPCF # add to sum over players in the defending team
            ptot = PPCFdef[i]+PPCFatt[i] # total pitch control probability 
            i += 1
        if i>=dT_array.size:
            print("Integration failed to converge: %1.3f" % (ptot) )
        return PPCFatt[i-1], PPCFdef[i-1]

def default_spearman_model_params(time_to_control_veto=3):
    """
    Returns the default parameters that define and evaluate the model. See Spearman 2018 for more details.
    
    Parameters
    -----------
    time_to_control_veto: If the probability that another team or player can get to the ball and control it is less than 10^-time_to_control_veto, ignore that player.
    
    
    Returns
    -----------
    
    params: dictionary of parameters required to determine and calculate the model
    
    """
    # key parameters for the model, as described in Spearman 2018
    params = {}
    # model parameters
    params['max_player_accel'] = 7. # maximum player acceleration m/s/s, not used in this implementation
    params['max_player_speed'] = 5. # maximum player speed m/s
    params['reaction_time'] = 0.7 # seconds, time taken for player to react and change trajectory. Roughly determined as vmax/amax
    params['tti_sigma'] = 0.45 # Standard deviation of sigmoid function in Spearman 2018 ('s') that determines uncertainty in player arrival time
    params['kappa_def'] =  1. # kappa parameter in Spearman 2018 (=1.72 in the paper) that gives the advantage defending players to control ball, I have set to 1 so that home & away players have same ball control probability
    params['lambda_att'] = 4.3 # ball control parameter for attacking team
    params['lambda_def'] = 4.3 * params['kappa_def'] # ball control parameter for defending team
    params['lambda_gk'] = params['lambda_def']*3.0 # make goal keepers must quicker to control ball (because they can catch it)
    params['average_ball_speed'] = 15. # average ball travel speed in m/s
    # numerical parameters for model evaluation
    params['int_dt'] = 0.04 # integration timestep (dt)
    params['max_int_time'] = 10 # upper limit on integral time
    params['model_converge_tol'] = 0.01 # assume convergence when PPCF>0.99 at a given location.
    # The following are 'short-cut' parameters. We do not need to calculated PPCF explicitly when a player has a sufficient head start. 
    # A sufficient head start is when the a player arrives at the target location at least 'time_to_control' seconds before the next player
    params['time_to_control_att'] = time_to_control_veto*np.log(10) * (np.sqrt(3)*params['tti_sigma']/np.pi + 1/params['lambda_att'])
    params['time_to_control_def'] = time_to_control_veto*np.log(10) * (np.sqrt(3)*params['tti_sigma']/np.pi + 1/params['lambda_def'])
    return params

class SpearmanPitchControl:
    def __init__(self, params=None):
        if params == None:
            self.params = default_spearman_model_params()
        else:
            self.params = params
        self.grid = pd.DataFrame([[i/1.05, j/0.68] for i in range(106) for j in range(69)], columns=['x','y'])

    def predict(self, xy):
        attacking_team = [spearman_player(p.player,p.x,p.y,p.dx,p.dy,self.params) for _, p in xy[xy.team == 'attack'].iterrows()]
        defending_team = [spearman_player(p.player,p.x,p.y,p.dx,p.dy,self.params) for _, p in xy[xy.team == 'defense'].iterrows()]
        ball = xy[xy.team.isna()].iloc[0][['x','y']].astype(float).to_numpy()

        pc_results = []
        for _, row in self.grid.iterrows():
            grid_point = row[['x', 'y']].astype(float).to_numpy()
            pc_results.append(calculate_spearman_pitch_control_at_target(grid_point, attacking_team, defending_team, ball, self.params))
        self.grid['control'] = np.array(pc_results)[:,0]

        return self.grid['control'] / self.grid['control'].max()

# Javier version by 
def influence_function(locs_home, locs_away, locs_ball, t, player_index, location, time_index, home_or_away):
  if home_or_away == 'h':
    data = locs_home.copy()
  elif home_or_away == 'a':
    data = locs_away.copy()
  else:
    raise ValueError("Enter either 'h' or 'a'.")
  # Added condition to process last frame
  if (time_index + 1) >= len(data[player_index]):
    time_index -= 1
  if np.all(np.isfinite(data[player_index][[time_index,time_index + 1],:])) & np.all(np.isfinite(locs_ball[0][time_index,:])):
    jitter = 1e-10 ## to prevent identically zero covariance matrices when velocity is zero
    ## compute velocity by fwd difference
    s = np.linalg.norm(data[player_index][time_index + 1,:] - data[player_index][time_index,:] + jitter) / (t[time_index + 1] - t[time_index])
    ## velocities in x,y directions
    sxy = (data[player_index][time_index + 1,:] - data[player_index][time_index,:] + jitter) / (t[time_index + 1] - t[time_index])
    ## angle between velocity vector & x-axis
    theta = np.arccos(sxy[0] / np.linalg.norm(sxy))
    ## rotation matrix
    R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    mu = data[player_index][time_index,:] + sxy * 0.5
    Srat = (s / 13) ** 2
    Ri = np.linalg.norm(locs_ball[0][time_index,:] - data[player_index][time_index,:])
    Ri = np.minimum(4 + Ri**3/ (18**3/6),10) ## don't think this function is specified in the paper but looks close enough to fig 9
    S = np.array([[(1 + Srat) * Ri / 2, 0], [0, (1 - Srat) * Ri / 2]])
    Sigma = np.matmul(R,S)
    Sigma = np.matmul(Sigma,S)
    Sigma = np.matmul(Sigma,np.linalg.inv(R)) ## this is not efficient, forgive me.
    out = mvn.pdf(location,mu,Sigma) / mvn.pdf(data[player_index][time_index,:],mu,Sigma)
  else:
    out = np.zeros(location.shape[0])
  return out

class FernandezPitchControl:
    def __init__(self):
        self.grid = pd.DataFrame([[i/1.05, j/0.68] for i in range(106) for j in range(69)], columns=['x','y'])
        self.yy, self.xx = np.meshgrid(np.linspace(0,68,69), np.linspace(0,105,106))

    def predict(self, xy, df):
        locs_home = [df[df.player==player_id][['x', 'y']].astype(float).to_numpy() * np.array([1.05,.68]) for player_id in df[df.team == 'attack'].player.unique()]
        locs_away = [df[df.player==player_id][['x', 'y']].astype(float).to_numpy() * np.array([1.05,.68]) for player_id in df[df.team == 'defense'].player.unique()]
        locs_ball = [df[df.team.isna()][['x', 'y']].astype(float).to_numpy() * np.array([1.05,.68])]
        t = pd.Series([i * 0.04 for i in range(len(locs_home[0]))])
        
        Zh = np.zeros(106*69)
        Za = np.zeros(106*69)
        for k in range(len(locs_home)):
            Zh += influence_function(locs_home, locs_away, locs_ball, t, k,np.c_[self.xx.flatten(),self.yy.flatten()],xy.frame.iloc[0],'h')
        for k in range(len(locs_away)):
            Za += influence_function(locs_home, locs_away, locs_ball, t, k,np.c_[self.xx.flatten(),self.yy.flatten()],xy.frame.iloc[0],'a')
        Zh = Zh.reshape((106,69))
        Za = Za.reshape((106,69))

        self.grid['control'] = 1 - (1 / (1 + np.exp(-Za + Zh))).flatten()

        return self.grid['control']# / self.grid['control'].max()







