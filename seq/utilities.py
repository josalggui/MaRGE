def change_axes(self):

    axes_x:int=self.axes[0]
    axes_y:int=self.axes[1]
    axes_z:int=self.axes[2]

# Change the axes
    if axes_x==1 and axes_y==2 and axes_z==3:
        x='readout'
        y='phase'
        z='slice'
        n_rd=self.n[0]
        n_ph=self.n[1]
        n_sl=self.n[2]
        fov_rd=self.fov[0]
        fov_ph=self.fov[1]
        fov_sl=self.fov[2]
    elif axes_x==1 and axes_y==3 and axes_z==2:
        x='readout'
        y='slice'
        z='phase'
        n_rd=self.n[0]
        n_ph=self.n[2]
        n_sl=self.n[1]
        fov_rd=self.fov[0]
        fov_ph=self.fov[2]
        fov_sl=self.fov[1]
    elif axes_x==2 and axes_y==1 and axes_z==3:
        x='phase'
        y='readout'
        z='slice'
        n_rd=self.n[1]
        n_ph=self.n[0]
        n_sl=self.n[2]
        fov_rd=self.fov[1]
        fov_ph=self.fov[0]
        fov_sl=self.fov[2]
    elif axes_x==3 and axes_y==1 and axes_z==2:
        x='slice'
        y='readout'
        z='phase' 
        n_rd=self.n[1]
        n_ph=self.n[2]
        n_sl=self.n[0] 
        fov_rd=self.fov[1]
        fov_ph=self.fov[2]
        fov_sl=self.fov[0]      
    elif axes_x==3 and axes_y==2 and axes_z==1:
        x='slice'
        y='phase'
        z='readout' 
        n_rd=self.n[2]
        n_ph=self.n[1]
        n_sl=self.n[0]
        fov_rd=self.fov[2]
        fov_ph=self.fov[1]
        fov_sl=self.fov[0]
    elif axes_x==2 and axes_y==3 and axes_z==1:
        x='phase'
        y='slice'
        z='readout'      
        n_rd=self.n[2]
        n_ph=self.n[0]
        n_sl=self.n[1] 
        fov_rd=self.fov[2]
        fov_ph=self.fov[0]
        fov_sl=self.fov[1]
    
    return x, y, z, n_rd, n_ph, n_sl, fov_rd, fov_ph, fov_sl
