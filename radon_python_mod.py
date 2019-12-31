import numpy.ma as ma
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
import cv2
import numpy as np
import math
from bresenham import bresenham
import matplotlib.pyplot as plt

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]
def gauss(x,a,x0,sigma,offset):
    return a*exp(-(x-x0)**2/(2*sigma**2))+offset
 
def ndim_grid(start,stop):
    # Set number of dimensions
    ndims = len(start)

    # List of ranges across all dimensions
    L = [np.arange(start[i],stop[i]) for i in range(ndims)]

    # Finally use meshgrid to form all combinations corresponding to all 
    # dimensions and stack them as M x ndims array
    return np.hstack((np.meshgrid(*L))).swapaxes(0,1).reshape(ndims,-1).T


def radon(vel_field, vel_field_e, n_p, n_theta, r_e, factor, plot):
    import numpy as np
    """
    This section performs the radon transform, from Stark et al. 2018.
    
    It is a long calculation, because it first calculates the Absolute Radon Transform (R_AB) of the velocity field, 
    then it iterates for multiple choices of the kinematic center. It determines the kinematic center by minimizing
    the asymmetry, A, of the Radon profile calculated from R_AB using a centroiding method.
    
    If the kinematic center is on the edge of the search grid of spaxels, it throws a flag and the code will rerun
    this function after expanding the grid.
    
    """
    #It first converts the x and y coordinates into p and theta coordinates (circular)
    #p is rho, which is the distance of the point on the velocity field from the kinematic center
    p_list = np.linspace(-int(np.shape(vel_field)[0]/2)+5,int(np.shape(vel_field)[0]/2)-5, int(np.shape(vel_field)[0]/2)+1)#was 5
    p_list = np.linspace(-int(np.shape(vel_field)[0]/2)+10, int(np.shape(vel_field)[0]/2)-10,n_p)#was 20
   
    #theta is the angle from the negative y axis CCW to the point on the velocity map.
    theta_list = np.linspace(0, 180, n_theta)#was 10
    
    #It searches over a grid of coordinates around the photometric center to find the 'kinematic center',
    #so it creates a 3x3 grid (from -3 to 3 including 0) in 2D
    box_list=list(ndim_grid([-3, -3],[4,4]))
    #If the kinematic center is not found on the first iteration, it expands the dimensions of the grid
    #by a factor of 2 upon rerunning.
    box_list=[factor*x for x in box_list]
    
    
    
    #Here I create a X and Y meshgrid type list of these points, since this is a rough approximation of the 
    #full grid of points in order to later plot what is happening.
    X_list=[]
    Y_list=[]

    for j in range(len(box_list)):

        X_list.append(int(np.shape(vel_field)[0]/2+box_list[j][0]))#-10+box_list[b][0])
        Y_list.append(int(np.shape(vel_field)[1]/2+box_list[j][1]))#-10+box_list[b][1])
        
    #print('X_list', X_list)
    #print('Y_list', Y_list)
    
    #creates empty lists that will be populated with R_AB, and all other derived quantities from this for every 
    #rho, theta point.
    
    
    #First run it for just the center one, index b=24 in order to normalize relative to the center
    #later on in the calculation of A.
    R_AB=[]
    R_AB_e=[]
    b=24
    
    plt.clf()
    plt.imshow(vel_field, cmap='RdBu_r')
    
    for i in range(len(p_list)):
        for j in range(len(theta_list)):
            #
            X = int(p_list[i]*math.cos(math.radians(theta_list[j]))+np.shape(vel_field)[0]/2+box_list[b][0])#-10+box_list[b][0])
            Y = int(p_list[i]*math.sin(math.radians(theta_list[j]))+np.shape(vel_field)[1]/2+box_list[b][1])#-10+box_list[b][1])
            

            '''We have an X and a Y and a theta (slope) so we should be able to get the intercept'''
            '''And then two more points on either end'''


            '''But we only want to calculate for things that are on the circle'''

            try:
                #if this point exists in the velocity field then you can continue
                test_value = vel_field[X,Y]
            except IndexError:
                R_AB.append(-1000)
                R_AB_e.append(-1000)
                continue
           

            if str(vel_field[X,Y]) == '--':
                R_AB.append(-1000)
                R_AB_e.append(-1000)
                continue
            #calculate the slope of the line segment from the kinematic center (in this case the photometric center)
            #to the given point
            deltay = Y - np.shape(vel_field)[1]/2
            deltax = X - np.shape(vel_field)[0]/2
            #draw a line perpendicular to this; the radon transform will be calculated along this line
            slope_p = math.tan(math.radians(theta_list[j]+90))#-deltax/deltay
            #draw a line from the point to where it intersects the bottom left of the map, which is the new origin
            intercept = Y - slope_p*X



            if slope_p > 1000:
                #vertical, so calculate along one value of X for the entire length of y
                x_min = X
                x_max = X
                y_min = 0
                y_max = np.shape(vel_field)[0]
            else:
                x_min = 0
                x_max = np.shape(vel_field)[0]
                y_min = intercept
                y_max = intercept+slope_p*x_max

            #This neat line draws a line through a given set of coordinates
            bres_list = list(bresenham(int(x_min), int(y_min), int(x_max), int(y_max)))

            #to calculate the absolute Bounded Radon Transform, do this for all points that are within r_e/2 of the center
            #of the line:
            vel_append=[]
            vel_append_e=[]
            within_r_e=[]
            for k in range(len(bres_list)):
                if bres_list[k][0] < 0 or bres_list[k][1] < 0:
                    
                    continue
                if np.sqrt((bres_list[k][0]-X)**2+(bres_list[k][1]-Y)**2) > r_e/2:
                    
                    continue
               
                try:
                    within_r_e.append(bres_list[k])
                    #print('distance from xcen', bres_list[k][0]-X)
                    #print('distance from ycen', bres_list[k][1]-Y)
                    #print('r_e/2', r_e/2)
                    #print('appending with this', bres_list[k][1], bres_list[k][0], vel_field[bres_list[k][1], bres_list[k][0]])
                    vel_append.append(vel_field[bres_list[k][1], bres_list[k][0]])
                    vel_append_e.append(vel_field_e[bres_list[k][1], bres_list[k][0]])
                except IndexError:
                   
                    continue
            
            
            xs_bres = [within_r_e[0][0],within_r_e[-1][0]]
            ys_bres = [within_r_e[0][1], within_r_e[-1][1]]
            #for b in range(len(bres_list)):
            #    plt.scatter(bres_list[b][0], bres_list[b][1])
            
            
            #clean it up, no masked values in here!
            vel_append_clean=[]
            vel_append_clean_e=[]
            for k in range(len(vel_append)):
                if ma.is_masked(vel_append[k]):
                    
                    continue
                else:
                    vel_append_clean.append(vel_append[k])
                    vel_append_clean_e.append(vel_append_e[k])

            if np.all(vel_append_clean)==0.0:
                R_AB.append(-1000)
                R_AB_e.append(-1000)
                continue
                
                
                
            indices=[] # Your output list
            for elem in set(vel_append_clean):
                   indices.append(vel_append_clean.index(elem))
            
            vel_append_clean = list(np.array(vel_append_clean)[indices])
            vel_append_clean_e = list(np.array(vel_append_clean_e)[indices])
            

            
            
            if vel_append_clean:
                
                vel_append_clean_masked = ma.masked_where(np.isinf(vel_append_clean_e), vel_append_clean)
                vel_append_clean_masked = ma.masked_where(vel_append_clean_masked==0.0, vel_append_clean_masked)
                vel_append_clean_masked_e = ma.masked_where(np.isinf(vel_append_clean_e), vel_append_clean_e)
                vel_append_clean_masked_e = ma.masked_where(vel_append_clean_masked==0.0, vel_append_clean_masked_e)
                inside=vel_append_clean_masked-np.mean(vel_append_clean_masked)
                
                R_AB.append(ma.sum(np.abs(inside)))
                
                # First calculate what the error is on the mean
                # which is the sum of every individual error divided by the vel
                error_mean = ma.mean(vel_append_clean_masked)*np.sqrt(ma.sum((vel_append_clean_masked_e/vel_append_clean_masked)**2))
                
                R_AB_error = np.sqrt(3*error_mean**2+ma.sum(vel_append_clean_masked_e)**2)
                
                R_AB_e.append(R_AB_error)
            else:
                R_AB.append(-1000)
                R_AB_e.append(-1000)
                continue
                
            # make these green if np.sum(np.abs(inside)) is less than some value
            if np.sum(np.abs(inside)) < 200:
             plt.plot(xs_bres, ys_bres, color='green', zorder=100)
             #plt.scatter(xs_bres, ys_bres, marker='^', color='white')
            else:
             plt.plot(xs_bres, ys_bres, color='white')
             #plt.scatter(xs_bres, ys_bres, marker='^', color='white')
            #plt.annotate('velocity sum '+str(round(np.sum(np.abs(inside)),1)), xy=(0.1,0.3), xycoords='axes fraction')
            #plt.annotate('p and theta '+str(round(p_list[i],1))+','+str(round(theta_list[j],1)), xy=(0.1,0.1), xycoords='axes fraction')
            #plt.annotate('X Y '+str(X)+','+str(Y), xy=(0.1,0.2), xycoords='axes fraction')
            
            #plt.xlim([0, np.shape(vel_field)[1]])
    plt.xlim([0, np.shape(vel_field)[1]])
    plt.ylim([np.shape(vel_field)[1],0])
    #ax.set_aspect('equal')
    plt.show()
    
   
    
    R_AB=ma.masked_where(np.isnan(R_AB), R_AB)
    R_AB = ma.masked_where(R_AB==-1000, R_AB)
    
    R_AB_e=ma.masked_where(np.isnan(R_AB), R_AB_e)
    R_AB_e = ma.masked_where(R_AB==-1000, R_AB_e)
    
    R_AB_array = np.reshape(R_AB, (len(p_list),len(theta_list)))
    R_AB_array_e = np.reshape(R_AB_e, (len(p_list),len(theta_list)))
    
    plt.clf()
    plt.imshow(R_AB_array)
    plt.colorbar()
    plt.show()
    
    
    
    #Now, extract the R_AB value at each rho value across all theta values.
    #This creates the Radon profile; the estimated value of theta hat that minimizes R_AB at each value of rho.
    #We minimize R_AB because we want the theta value at each rho that 

    #these are the estimated values of theta that are best fit for a value of rho
    theta_hat=[]
    theta_hat_e=[]

    for l in range(len(p_list)):

        marginalized = R_AB_array[l,:]
        marginalized_e = R_AB_array_e[l,:]
        marginalized = ma.masked_where(marginalized<1e-4, marginalized)
        marginalized_e = ma.masked_where(marginalized<1e-4, marginalized_e)
        count = len([i for i in marginalized if i > 1e-3])
        #count up how many elements are in the row of R_AB --> if it is less than 6, don't measure it
        #because it will cause an error when trying to fit a Gaussian, k = n+1
        if count < 6:
            theta_hat.append(0)
            theta_hat_e.append(0)
            continue

        if ma.is_masked(marginalized)==True:
            theta_hat.append(0)
            theta_hat_e.append(0)
            continue

        try:
            #initially, try to fit a negative gaussian to determine theta hat
            popt,pcov = curve_fit(gauss,theta_list,marginalized,sigma = marginalized_e,p0=[-abs(np.min(marginalized)-np.max(marginalized)),theta_list[list(marginalized).index(np.min(marginalized))],20,np.mean(marginalized)])
            append_value = popt[1]
            
            
            
        except RuntimeError or OptimizeError: 
            theta_hat.append(0)
            theta_hat_e.append(0)

            continue
            
            

        #if the code fits a positive Gaussian, try a new guess for the correct theta hat, the second smallest value       
        if (popt[0]>0):
            try:
                popt,pcov = curve_fit(gauss,theta_list,marginalized,sigma=marginalized_e,p0=[-abs(np.min(marginalized)-np.max(marginalized)),theta_list[list(marginalized).index(sorted(marginalized)[1])],20,np.mean(marginalized)])
                append_value = popt[1]
                if popt[0] > 0:
                    #if this doesn't work, quit and move on

                    theta_hat.append(0)
                    theta_hat_e.append(0)

                    continue

                
                

            except RuntimeError or OptimizeError :
                theta_hat.append(0)
                theta_hat_e.append(0)

                continue
        
        #sometimes, it is necessary to shift the xs because the peak is at 180, which is right at the edge
        if ((popt[1] - 3*np.sqrt(pcov[1][1])) < 0) or ((popt[1] + 3*np.sqrt(pcov[1][1])) > 180):
            theta_list_shifted = theta_list+find_nearest(theta_list,90)
            index_1 = list(theta_list).index(find_nearest(theta_list,90))
            new_marginalized = np.concatenate((marginalized[index_1:], marginalized[:index_1]))
            new_marginalized_e = np.concatenate((marginalized_e[index_1:], marginalized_e[:index_1]))

            try:
                popt,pcov = curve_fit(gauss,theta_list_shifted,new_marginalized,sigma = new_marginalized_e, p0=[-abs(np.min(new_marginalized)-np.max(new_marginalized)),theta_list_shifted[list(new_marginalized).index(np.min(new_marginalized))],20,np.mean(new_marginalized)])
                

                if popt[0] > 0:
                    theta_hat.append(0)
                    theta_hat_e.append(0)

                    continue
                    
                    
                if popt[1] > 180:
                    append_value = popt[1]-180
                else:
                    append_value = popt[1]

            except RuntimeError or OptimizeError :
                theta_hat.append(0)
                theta_hat_e.append(0)

                continue
            
        theta_hat.append(append_value)
        theta_hat_e.append(np.sqrt(pcov[1][1]))

    #now to calculate A, it is necessary to sum the values of theta hat at a mirror image of themselves
    #across p=0
    delta_theta_sum=[]
    delta_theta_sum_e=[]


    for l in range(int(len(p_list)/2)):
        if (theta_hat[0+l]==0) or (theta_hat[-1-l]==0) or (abs(theta_hat[0+l])>180) or (abs(theta_hat[-1-l]) > 180):

            delta_theta_sum.append(0)
            delta_theta_sum_e.append(0)
        else:
            if abs(theta_hat[0+l]-theta_hat[-1-l]) > 90:
                #because the maximum you can be apart is 90
                inside = 180 - abs(theta_hat[0+l]-theta_hat[-1-l])
            else:
                inside = abs(theta_hat[0+l]-theta_hat[-1-l])

            delta_theta_sum.append(inside)
            #I would also like to have an error estimate on this quantity:
            delta_theta_sum_e.append(np.sqrt((theta_hat_e[0+l])**2+
                                         (theta_hat_e[-1-l])**2))
    


    delta_theta_sum_masked=ma.masked_where(np.array(delta_theta_sum)==0, delta_theta_sum)

    

    OG_weight=ma.count(delta_theta_sum_masked)
    
    
    plt.clf()
    fig=plt.figure()
    ax0 = fig.add_subplot(111)
    if plot=='yes':
        
        im0 = ax0.imshow(vel_field, cmap='RdBu_r')
    
    
    #Okay now do this for all the other positions in box_list
    
            
    A_list=[]
    A_e_list=[]
    A_2_list=[]
    R_AB_list=[]
    R_AB_list_e=[]
    theta_hat_list=[]
    theta_hat_e_list=[]
    
    X_list_real=[]
    Y_list_real=[]
    

    for b in range(len(box_list)):
        R_AB=[]
        R_AB_e=[]
        X_list_real.append(np.shape(vel_field)[0]/2+box_list[b][0])
        Y_list_real.append(np.shape(vel_field)[1]/2+box_list[b][1])
        for i in range(len(p_list)):
            for j in range(len(theta_list)):
                #
                X = int(p_list[i]*math.cos(math.radians(theta_list[j]))+np.shape(vel_field)[0]/2+box_list[b][0])#-10+box_list[b][0])
                Y = int(p_list[i]*math.sin(math.radians(theta_list[j]))+np.shape(vel_field)[1]/2+box_list[b][1])#-10+box_list[b][1])
                
                try:
                    #if this point exists in the velocity field then you can continue
                    test_value = vel_field[X,Y]
                except IndexError:
                    R_AB.append(-1000)
                    R_AB_e.append(-1000)
                    continue

                if str(vel_field[X,Y]) == '--':
                    R_AB.append(-1000)
                    R_AB_e.append(-1000)
                    continue
                #calculate the slope of the line segment from the kinematic center (in this case the photometric center)
                #to the given point
                deltay = Y - np.shape(vel_field)[1]/2
                deltax = X - np.shape(vel_field)[0]/2
                #draw a line perpendicular to this; the radon transform will be calculated along this line
                slope_p = math.tan(math.radians(theta_list[j]+90))#-deltax/deltay
                #draw a line from the point to where it intersects the bottom left of the map, which is the new origin
                intercept = Y - slope_p*X



                if slope_p > 1000:
                    #vertical, so calculate along one value of X for the entire length of y
                    x_min = X
                    x_max = X
                    y_min = 0
                    y_max = np.shape(vel_field)[0]
                else:
                    x_min = 0
                    x_max = np.shape(vel_field)[0]
                    y_min = intercept
                    y_max = intercept+slope_p*x_max

                #This neat line draws a line through a given set of coordinates
                bres_list = list(bresenham(int(x_min), int(y_min), int(x_max), int(y_max)))

                #to calculate the absolute Bounded Radon Transform, do this for all points that are within r_e/2 of the center
                #of the line:
                vel_append=[]
                vel_e_append=[]
                for k in range(len(bres_list)):
                    if bres_list[k][0] < 0 or bres_list[k][1] < 0:
                        continue
                    if np.sqrt((bres_list[k][0]-X)**2+(bres_list[k][1]-Y)**2) > r_e/2:
                        continue

                    try:

                        vel_append.append(vel_field[bres_list[k][1], bres_list[k][0]])
                        vel_e_append.append(vel_field_e[bres_list[k][1], bres_list[k][0]])
                        #vel_new[bres_list[j][1], bres_list[j][0]] = vel_field[bres_list[j][1], bres_list[j][0]]
                    except IndexError:
                        continue
                
                
                
                #clean it up, no masked values in here!
                
                vel_append_clean=[]
                vel_append_clean_e=[]
                for k in range(len(vel_append)):
                    if ma.is_masked(vel_append[k]):
                        continue
                    else:
                        vel_append_clean.append(vel_append[k])
                        vel_append_clean_e.append(vel_append_e[k])
                        
                        
                if np.all(vel_append_clean)==0.0:
                    R_AB.append(-1000)
                    R_AB_e.append(-1000)
                    continue
                # Next, delete all repeated indices from both arrays - this is to avoid overusing voronoi
                # bins so that the method isn't so overly dependent on the outskirts where there are larger Voronoi bins
                    
           
                indices=[] # Your output list
                for elem in set(vel_append_clean):
                       indices.append(vel_append_clean.index(elem))
                
                vel_append_clean = list(np.array(vel_append_clean)[indices])
                vel_append_clean_e = list(np.array(vel_append_clean_e)[indices])
                
                
                #finally, create R_AB by summing all of these velocity differences
                if vel_append_clean:
                    #try:
                    vel_append_clean_masked = ma.masked_where(np.isinf(vel_append_clean_e), vel_append_clean)
                    #except IndexError:
                    #    R_AB.append(-1000)
                    #    R_AB_e.append(-1000)
                    #    continue
                    vel_append_clean_masked = ma.masked_where(vel_append_clean_masked==0.0, vel_append_clean_masked)
                    vel_append_clean_masked_e = ma.masked_where(np.isinf(vel_append_clean_e), vel_append_clean_e)
                    vel_append_clean_masked_e = ma.masked_where(vel_append_clean_masked==0.0, vel_append_clean_masked_e)
                    inside=vel_append_clean_masked-np.mean(vel_append_clean_masked)
                    R_AB.append(ma.sum(np.abs(inside)))
                    
                    # First calculate what the error is on the mean
                    # which is the sum of every individual error divided by the vel
                    error_mean = ma.mean(vel_append_clean_masked)*np.sqrt(ma.sum((vel_append_clean_masked_e/vel_append_clean_masked)**2))
                    
                    R_AB_error = np.sqrt(3*error_mean**2+ma.sum(vel_append_clean_masked_e)**2)
                    
                    R_AB_e.append(R_AB_error)
                else:
                    R_AB.append(-1000)
                    R_AB_e.append(-1000)
                    continue


        R_AB=ma.masked_where(np.isnan(R_AB), R_AB)
        R_AB = ma.masked_where(R_AB==-1000, R_AB)
        R_AB_e = ma.masked_where(R_AB_e==-1000, R_AB_e)

        R_AB_array = np.reshape(R_AB, (len(p_list),len(theta_list)))
        R_AB_array_e = np.reshape(R_AB_e, (len(p_list),len(theta_list)))


        

        #Now, extract the R_AB value at each rho value across all theta values.
        #This creates the Radon profile; the estimated value of theta hat that minimizes R_AB at each value of rho.
        #We minimize R_AB because we want the theta value at each rho that 

        #these are the estimated values of theta that are best fit for a value of rho
        theta_hat=[]
        theta_hat_e=[]

        for l in range(len(p_list)):

            marginalized = R_AB_array[l,:]
            marginalized_e = R_AB_array_e[l,:]
            marginalized = ma.masked_where(marginalized<1e-4, marginalized)
            count = len([i for i in marginalized if i > 1e-3])
            #count up how many elements are in the row of R_AB --> if it is less than 6, don't measure it
            #because it will cause an error when trying to fit a Gaussian, k = n+1
            if count < 6:
                theta_hat.append(0)
                theta_hat_e.append(0)
                continue

            if ma.is_masked(marginalized)==True:
                theta_hat.append(0)
                theta_hat_e.append(0)
                continue

            try:
                popt,pcov = curve_fit(gauss,theta_list,marginalized,sigma=marginalized_e,p0=[-abs(np.min(marginalized)-np.max(marginalized)),theta_list[list(marginalized).index(np.min(marginalized))],20,np.mean(marginalized)])
                append_value = popt[1]

                

            #sometimes, it is necessary to shift the xs because the peak is at 180, which is right at the cutoff
            except RuntimeError or OptimizeError: 
                theta_hat.append(0)
                theta_hat_e.append(0)

                continue
                


                    
            if (popt[0]>0):
                try:
                    popt,pcov = curve_fit(gauss,theta_list,marginalized,sigma=marginalized_e,p0=[-abs(np.min(marginalized)-np.max(marginalized)),theta_list[list(marginalized).index(sorted(marginalized)[1])],20,np.mean(marginalized)])
                    append_value = popt[1]
                    if popt[0] > 0:


                        theta_hat.append(0)
                        theta_hat_e.append(0)

                        continue


                except RuntimeError or OptimizeError :
                    theta_hat.append(0)
                    theta_hat_e.append(0)

                    continue
            if ((popt[1] - 3*np.sqrt(pcov[1][1])) < 0) or ((popt[1] + 3*np.sqrt(pcov[1][1])) > 180):
                theta_list_shifted = theta_list+find_nearest(theta_list,90)
                index_1 = list(theta_list).index(find_nearest(theta_list,90))
                new_marginalized = np.concatenate((marginalized[index_1:], marginalized[:index_1]))
                new_marginalized_e = np.concatenate((marginalized_e[index_1:], marginalized_e[:index_1]))

                try:
                    popt,pcov = curve_fit(gauss,theta_list_shifted,new_marginalized,sigma=marginalized_e,p0=[-abs(np.min(new_marginalized)-np.max(new_marginalized)),theta_list_shifted[list(new_marginalized).index(np.min(new_marginalized))],20,np.mean(new_marginalized)])
                    
                    if popt[0] > 0:
                        theta_hat.append(0)
                        theta_hat_e.append(0)

                        continue

                    
                    if popt[1] > 180:
                        append_value = popt[1]-180
                    else:
                        append_value = popt[1]

                except RuntimeError or OptimizeError :
                    theta_hat.append(0)
                    theta_hat_e.append(0)

                    continue

            theta_hat.append(append_value)
            theta_hat_e.append(np.sqrt(pcov[1][1]))



 

        delta_theta_sum=[]
        delta_theta_sum_e=[]
        
        
        for l in range(int(len(p_list)/2)):
            if (theta_hat[0+l]==0) or (theta_hat[-1-l]==0) or (abs(theta_hat[0+l])>180) or (abs(theta_hat[-1-l]) > 180):
                
                delta_theta_sum.append(0)
                delta_theta_sum_e.append(0)
            else:
                
                
                if abs(theta_hat[0+l]-theta_hat[-1-l]) > 90:
                    #because the maximum you can be apart is 90
                    inside = 180 - abs(theta_hat[0+l]-theta_hat[-1-l])
                else:
                    inside = abs(theta_hat[0+l]-theta_hat[-1-l])
                
                delta_theta_sum.append(inside)
                delta_theta_sum_e.append(np.sqrt((theta_hat_e[0+l])**2+
                                             (theta_hat_e[-1-l])**2))
        
            
            
            
        theta_hat_mask = ma.masked_where(theta_hat==0, theta_hat)
        theta_hat_mask = ma.masked_where(abs(theta_hat_mask) >180, theta_hat_mask)
        
        theta_hat_list.append(theta_hat_mask)
        theta_hat_e_list.append(theta_hat_e)
        
        delta_theta_sum_masked=ma.masked_where(np.array(delta_theta_sum)==0, delta_theta_sum)

        #A is weighted by the A value of the center of the map
        A = ((ma.sum(delta_theta_sum)/(ma.count(delta_theta_sum_masked)**2)))*OG_weight
       
        A_percent_e = []
        for l in range(len(delta_theta_sum_e)):
            if delta_theta_sum[l] != 0 :
                A_percent_e.append((delta_theta_sum_e[l])**2)
        A_abs_error = np.sqrt(np.sum(A_percent_e))#so this is on the numerator of the A error only
        A_error = (A_abs_error/ma.sum(delta_theta_sum))*A
       
        
        
        A_list.append(A)
        
        
        A_e_list.append(A_error)

        #Also calculates the other types of asymmetry:
        #A_2
        delta_theta_sum=ma.masked_where(np.array(delta_theta_sum)==0, delta_theta_sum)
        delta_theta_sum_e=ma.masked_where(np.array(delta_theta_sum_e)==0, delta_theta_sum_e)
        
        
        A_2 = ma.sum(delta_theta_sum/delta_theta_sum_e)
        
        
        
        A_2_list.append(A_2)
        
        R_AB_list.append(R_AB_array)
        R_AB_list_e.append(R_AB_array_e)
        
    
      
    
    print('A_list before masking', A_list)
    A_list = np.array(A_list)
    A_list = ma.masked_where(np.isnan(A_list), A_list)
    print('this is the shape of A_list', np.shape(A_list))
    print('this is the shape of box list', np.shape(box_list))
    #len(box_list)
    A_list_array = np.reshape(np.array(A_list), (7,7))
    A_e_list_array = np.reshape(np.array(A_e_list), (7,7))



    
    first_number = np.shape(vel_field)[0]/2-3*factor#33-3*factor#33-3*factor = 27 if factor = 2
    x = factor*np.linspace(0, np.shape(A_list_array)[0]-1,np.shape(A_list_array)[0])+first_number
    #y = (factor*6-factor*np.linspace(0, np.shape(A_list_array)[1]-1, np.shape(A_list_array)[1]))+first_number
    y = (factor*np.linspace(0, np.shape(A_list_array)[1]-1, np.shape(A_list_array)[1]))+first_number
    
    if min(X_list_real) != min(x):
        #this is improper scaling
        print('first number', first_number)
        print('x and y', x, y)
    
        print('this is what it should match X_list', X_list_real, Y_list_real)
    
        STOP
    
    
    x, y = np.meshgrid(x, y)
    
    
    
    
    
    cs = ax0.contour(x,y, A_list_array, 5, colors='orange')#was 5
    p = cs.levels#collections[0].get_paths()[0]
    
    if plot=='yes':
        im1 = ax0.scatter(np.flip(Y_list_real, axis=0),X_list_real,c=A_list, s=30, zorder=100)

        plt.colorbar(im0)
    
    #Now, snatch the last level and use it to blur everything else out and find the center
    #now fill with zeros and do your best to turn into a binary bitmask'''
    ret,thresh = cv2.threshold(A_list_array,p[-1]/2,100,cv2.THRESH_BINARY_INV)
    M = cv2.moments(thresh)



    # calculate x,y coordinate of center
    try:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    except ZeroDivisionError:
        print('A_list', A_list_array)
        print('levels', p)
        print('older p threshold', p[-1]/2)
        print('mean of A_list_array', np.mean(A_list_array))
        ret,thresh = cv2.threshold(A_list_array,np.mean(A_list_array),100,cv2.THRESH_BINARY_INV)
        M = cv2.moments(thresh)
        try:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        except:
            cX = 3
            cY = 3
    
    x_match = np.shape(vel_field)[0]/2 + factor*(3 - cX) #cX_flip+33-3*factor
    y_match = np.shape(vel_field)[0]/2 + factor*(cY - 3)
       
    
    
    
    
    
    
    if plot=='yes':
        

        print('this is the cX and cY', x_match, y_match)
        print('and this is half', np.shape(vel_field)[0]/2)
        print('this is og CX and cY', cX, cY)


        plt.scatter( x_match, y_match, marker='x', color='red', zorder=105)
        plt.axhline(y=np.shape(vel_field)[0]/2, color='red')
        plt.axvline(x=np.shape(vel_field)[0]/2, color='red')

        plt.show()
    
    
    
    
        plt.clf()
        plt.imshow(thresh)
        plt.scatter(cX,cY, marker='x', color='red')

        x = np.linspace(0, np.shape(A_list_array)[0]-1,np.shape(A_list_array)[0])
        y = (np.linspace(0, np.shape(A_list_array)[1]-1, np.shape(A_list_array)[1]))
        x, y = np.meshgrid(x,y)

        plt.contour(x,y, A_list_array, 5, colors='orange')#was 5
        plt.xlim([6,0])

        plt.show()
    
    
    
    
    
    '''now find the min index of this'''
    for ii in range(len(X_list)):
        if X_list[ii]==find_nearest(np.array(X_list),x_match) and Y_list[ii]==find_nearest(np.array(np.flip(Y_list, axis=0)),y_match):
            min_index = ii
        
    
    
    
    if factor ==1:
        if abs(box_list[min_index][0])>2 or abs(box_list[min_index][1])>2:
            #then we are on the edge
            expand=1
            # Just end it here
            return 0,0, 0,  0,0, 0, 0, 0, expand, 0
        else:
            expand=0
    else:
        if abs(box_list[min_index][0])>5 or abs(box_list[min_index][1])>5:
            min_index=24
        expand=0
    # Go back and get the x and y coordinates of the minimum index and make
    # sure it is drawing lines in the right place
    print('min box list', box_list, min_index, box_list[min_index])
    
    if plot=='yes':
    
       plt.clf()
       plt.imshow(vel_field, cmap='RdBu_r')
       box_list_min=box_list[min_index]
       
       
       xcen = int(np.shape(vel_field)[0]/2+box_list_min[0])#-10+box_list[b][0])
       ycen = int(np.shape(vel_field)[1]/2+box_list_min[1])#-10+box_list[b][1])
       print('xcen ycen', xcen, ycen)
       plt.scatter(xcen, ycen, marker='*', color='orange', edgecolor='orange')
       
       
       R_AB_fake = []
       for i in range(len(p_list)):
           for j in range(len(theta_list)):
               #
               
               X = int(p_list[i]*math.cos(math.radians(theta_list[j]))+np.shape(vel_field)[0]/2+box_list_min[0])#-10+box_list[b][0])
               Y = int(p_list[i]*math.sin(math.radians(theta_list[j]))+np.shape(vel_field)[1]/2+box_list_min[1])#-10+box_list[b][1])
               plt.scatter(X, Y, marker='*', color='white', edgecolor='black')
               
               
               try:
                   #if this point exists in the velocity field then you can continue
                   test_value = vel_field[X,Y]
               except IndexError:
                   R_AB_fake.append(-1000)
                   continue

               if str(vel_field[X,Y]) == '--':
                   R_AB_fake.append(-1000)
                   continue
               #calculate the slope of the line segment from the kinematic center (in this case the photometric center)
               #to the given point
               deltay = Y - np.shape(vel_field)[1]/2
               deltax = X - np.shape(vel_field)[0]/2
               #draw a line perpendicular to this; the radon transform will be calculated along this line
               slope_p = math.tan(math.radians(theta_list[j]+90))#-deltax/deltay
               #draw a line from the point to where it intersects the bottom left of the map, which is the new origin
               intercept = Y - slope_p*X


               if slope_p > 1000:
                   #vertical, so calculate along one value of X for the entire length of y
                   x_min = X
                   x_max = X
                   y_min = 0
                   y_max = np.shape(vel_field)[0]
               else:
                   x_min = 0
                   x_max = np.shape(vel_field)[0]
                   y_min = intercept
                   y_max = intercept+slope_p*x_max

               #This neat line draws a line through a given set of coordinates
               bres_list = list(bresenham(int(x_min), int(y_min), int(x_max), int(y_max)))

               

               #to calculate the absolute Bounded Radon Transform, do this for all points that are within r_e/2 of the center
               #of the line:
               vel_append=[]
               vel_e_append = []
               within_r_e = []
               for k in range(len(bres_list)):
                   if bres_list[k][0] < 0 or bres_list[k][1] < 0:
                       continue
                   if np.sqrt((bres_list[k][0]-X)**2+(bres_list[k][1]-Y)**2) > r_e/2:
                       continue

                   try:
                       within_r_e.append(bres_list[k])
                       vel_append.append(vel_field[bres_list[k][1], bres_list[k][0]])
                       vel_e_append.append(vel_field_e[bres_list[k][1], bres_list[k][0]])
                       #vel_new[bres_list[j][1], bres_list[j][0]] = vel_field[bres_list[j][1], bres_list[j][0]]
                   except IndexError:
                       continue

               #clean it up, no masked values in here!
               vel_append_clean_fake=[]
               vel_append_clean_fake_e=[]
               for k in range(len(vel_append)):
                   if ma.is_masked(vel_append[k]):
                       continue
                   else:
                       vel_append_clean_fake.append(vel_append[k])
                       vel_append_clean_fake_e.append(vel_append_e[k])
                       
               
               indices=[] # Your output list
               for elem in set(vel_append_clean_fake):
                      indices.append(vel_append_clean_fake.index(elem))
               
               vel_append_clean_fake = list(np.array(vel_append_clean_fake)[indices])
               vel_append_clean_fake_e = list(np.array(vel_append_clean_fake_e)[indices])
               

               #finally, create R_AB by summing all of these velocity differences
               if vel_append_clean_fake:
                   vel_append_clean_masked_fake = ma.masked_where(np.isinf(vel_append_clean_fake_e), vel_append_clean_fake)
                   vel_append_clean_masked_fake = ma.masked_where(vel_append_clean_masked_fake==0.0, vel_append_clean_masked_fake)
                   vel_append_clean_masked_fake_e = ma.masked_where(np.isinf(vel_append_clean_fake_e), vel_append_clean_fake_e)
                   vel_append_clean_masked_fake_e = ma.masked_where(vel_append_clean_masked_fake==0.0, vel_append_clean_masked_fake_e)
                   inside=vel_append_clean_masked_fake-np.mean(vel_append_clean_masked_fake)
                   R_AB_fake.append(ma.sum(np.abs(inside)))
               else:
                   R_AB_fake.append(-1000)
                   continue

               
               #print(vel_append_clean_fake)
               #print(np.sum(np.abs(inside)))
               xs_bres = [within_r_e[0][0],within_r_e[-1][0]]
               ys_bres = [within_r_e[0][1], within_r_e[-1][1]]
               #for b in range(len(bres_list)):
               #    plt.scatter(bres_list[b][0], bres_list[b][1])
               
               # make these green if np.sum(np.abs(inside)) is less than some value
               if np.sum(np.abs(inside)) < np.mean(R_AB_list[min_index])/2:
                plt.plot(xs_bres, ys_bres, color='green', zorder=100)
                #plt.scatter(xs_bres, ys_bres, marker='^', color='white')
               else:
                plt.plot(xs_bres, ys_bres, color='white')
                #plt.scatter(xs_bres, ys_bres, marker='^', color='white')
               #plt.annotate('velocity sum '+str(round(np.sum(np.abs(inside)),1)), xy=(0.1,0.3), xycoords='axes fraction')
               #plt.annotate('p and theta '+str(round(p_list[i],1))+','+str(round(theta_list[j],1)), xy=(0.1,0.1), xycoords='axes fraction')
               #plt.annotate('X Y '+str(X)+','+str(Y), xy=(0.1,0.2), xycoords='axes fraction')
       plt.xlim([0, np.shape(vel_field)[1]])
       plt.ylim([np.shape(vel_field)[1],0])
       #ax.set_aspect('equal')
       plt.show()
       
       
       
       R_AB_fake=ma.masked_where(np.isnan(R_AB_fake), R_AB_fake)
       R_AB_fake = ma.masked_where(R_AB_fake==-1000, R_AB_fake)
       R_AB_fake = ma.masked_where(R_AB_fake==0.0, R_AB_fake)

       R_AB_fake_array = np.reshape(R_AB_fake, (len(p_list),len(theta_list)))
       
      
    return box_list[min_index],R_AB_list[min_index], A_list[min_index],  A_2_list[min_index],p_list, theta_list, theta_hat_list[min_index], theta_hat_e_list[min_index], expand, R_AB_list_e[min_index]
