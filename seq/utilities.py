
def change_axes(self):

    axes_x=self.sequence.axes[0]
    axes_y=self.sequence.axes[1]
    axes_z=self.sequence.axes[2]

# Change the axes
    if axes_x==0 and axes_y==1 and axes_z==2:
        x='readout'
        y='phase'
        z='slice'
        n_rd=self.sequence.nPoints[0]
        n_ph=self.sequence.nPoints[1]
        n_sl=self.sequence.nPoints[2]
        fov_rd=self.sequence.fov[0]
        fov_ph=self.sequence.fov[1]
        fov_sl=self.sequence.fov[2]
    elif axes_x==0 and axes_y==2 and axes_z==1:
        x='readout'
        y='slice'
        z='phase'
        n_rd=self.sequence.nPoints[0]
        n_ph=self.sequence.nPoints[2]
        n_sl=self.sequence.nPoints[1]
        fov_rd=self.sequence.fov[0]
        fov_ph=self.sequence.fov[2]
        fov_sl=self.sequence.fov[1]
    elif axes_x==1 and axes_y==0 and axes_z==2:
        x='phase'
        y='readout'
        z='slice'
        n_rd=self.sequence.nPoints[1]
        n_ph=self.sequence.nPoints[0]
        n_sl=self.sequence.nPoints[2]
        fov_rd=self.sequence.fov[1]
        fov_ph=self.sequence.fov[0]
        fov_sl=self.sequence.fov[2]
    elif axes_x==2 and axes_y==0 and axes_z==1:
        x='slice'
        y='readout'
        z='phase' 
        n_rd=self.sequence.nPoints[1]
        n_ph=self.sequence.nPoints[2]
        n_sl=self.sequence.nPoints[0] 
        fov_rd=self.sequence.fov[1]
        fov_ph=self.sequence.fov[2]
        fov_sl=self.sequence.fov[0]      
    elif axes_x==2 and axes_y==1 and axes_z==0:
        x='slice'
        y='phase'
        z='readout' 
        n_rd=self.sequence.nPoints[2]
        n_ph=self.sequence.nPoints[1]
        n_sl=self.sequence.nPoints[0]
        fov_rd=self.sequence.fov[2]
        fov_ph=self.sequence.fov[1]
        fov_sl=self.sequence.fov[0]
    elif axes_x==1 and axes_y==2 and axes_z==0:
        x='phase'
        y='slice'
        z='readout'      
        n_rd=self.sequence.nPoints[2]
        n_ph=self.sequence.nPoints[0]
        n_sl=self.sequence.nPoints[1] 
        fov_rd=self.sequence.fov[2]
        fov_ph=self.sequence.fov[0]
        fov_sl=self.sequence.fov[1]
    
    return x, y, z, n_rd, n_ph, n_sl, fov_rd, fov_ph, fov_sl
