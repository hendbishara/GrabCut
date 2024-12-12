import numpy as np
import cv2
import argparse
from sklearn.mixture import GaussianMixture
import igraph


GC_BGD = 0 # Hard bg pixel
GC_FGD = 1 # Hard fg pixel, will not be used
GC_PR_BGD = 2 # Soft bg pixel
GC_PR_FGD = 3 # Soft fg pixel
#gloabl varibals
first_run,last_run=True, False
n_edges,n_weights,t_edges, edges=[],[],[],[]
K,prev_energy,lama,epsilon=0,0,1,1000
g = igraph.Graph()

# Define the GrabCut algorithm function
def grabcut(img, rect, n_iter=5):
    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect

    # Convert from absolute cordinates
    w -= x
    h -= y

    #Initalize the inner square to Foreground
    mask[y:y+h, x:x+w] = GC_PR_FGD
    mask[rect[1]+rect[3]//2, rect[0]+rect[2]//2] = GC_FGD

    bgGMM, fgGMM = initalize_GMMs(img, mask) 
    

    num_iters = 1000
    for i in range(num_iters):
        #Update GMM
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)
        
        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)
        
        mask = update_mask(mincut_sets, mask)
        
        if check_convergence(energy):
            break

    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM


def initalize_GMMs(img, mask):
    #initializing component index
    global lama
    # Reshape the image and mask
    h, w, c = img.shape
    img_reshaped = img.reshape((-1, c))
    mask_reshaped = mask.reshape(-1)

    # Extract foreground and background pixels
    foreground_pixels = img_reshaped[(mask_reshaped == GC_PR_FGD) | (mask_reshaped == GC_FGD)]
    background_pixels = img_reshaped[mask_reshaped == GC_BGD]
    hard_fore=img_reshaped[mask_reshaped == GC_FGD]

    #calculate the diffrence between foreground and hardground
    fresh=np.power((background_pixels.astype(np.int64) - hard_fore.astype(np.int64)),2)
    fresh=np.sum(fresh, axis=-1)
    fresh=np.sum(fresh)//background_pixels.shape[0]

    #check if average fresh is very small or very big
    if fresh <2000 or fresh >40000:
        lama=4

    #initialize GMM model
    bgGMM = GaussianMixture(n_components=3)
    fgGMM = GaussianMixture(n_components=3)
    back_means,back_covs,back_weigts=initialize_gmm_clusters(background_pixels, 3)
    fore_means,fore_covs,fore_weigts=initialize_gmm_clusters(foreground_pixels, 3)
    
    #build the GMM
    bgGMM, fgGMM=build_GMM(bgGMM,fgGMM,back_means,back_covs,back_weigts,fore_means,fore_covs,fore_weigts)
    return bgGMM, fgGMM


# Define helper functions for the GrabCut algorithm
def update_GMMs(img, mask, bgGMM, fgGMM):
    # TODO: implement GMM component assignment step
    #initializing component index
    global first_run
    
    #for the first run we skip the update to not repeate the build GMM
    if first_run:
        return bgGMM, fgGMM
    h, w, c = img.shape
    img_reshaped = img.reshape((-1, c))
    mask_reshaped = mask.reshape(-1)
    
    # Extract foreground and background pixels
    foreground_pixel = img_reshaped[(mask_reshaped == GC_PR_FGD) | (mask_reshaped == GC_FGD)]
    background_pixel = img_reshaped[(mask_reshaped == GC_PR_BGD) | (mask_reshaped == GC_BGD)]
    #assign each pixel to the most likely Guassian component
    back_like=bgGMM.predict(background_pixel)
    fore_like=fgGMM.predict(foreground_pixel)
    
    #update each component according to the new learned GMM
    back_means, back_covs, back_weigts=[],[],[]
    fore_means, fore_covs, fore_weigts=[],[],[]
    for comp in range(3):
        f = foreground_pixel[(fore_like == comp)]
        b = background_pixel[(back_like == comp)]
        
        back_means.append(np.mean(b,axis=0))
        back_covs.append(np.cov(b.T) + 1e-6 * np.eye(3))
        back_weigts.append((len(b)/len(background_pixel)))
            
    
        fore_means.append(np.mean(f,axis=0))
        fore_covs.append(np.cov(f.T) + 1e-6 * np.eye(3))
        fore_weigts.append((len(f)/len(foreground_pixel)))
    
    #build the GMM
    bgGMM, fgGMM=build_GMM(bgGMM,fgGMM,back_means,back_covs,back_weigts,fore_means,fore_covs,fore_weigts)
    return bgGMM, fgGMM


def calculate_mincut(img, mask, bgGMM, fgGMM):
    # TODO: implement energy (cost) calculation step and mincut
    global first_run, n_edges, n_weights, hello, t_edges, edges, g
    min_cut = [[], []]
    energy = 0
    rows, cols, c = img.shape
    #get index for foreground node and background node
    fore_node=rows*cols
    back_node=rows*cols+1
    if first_run:
        #this part run only once in the first run
        img = img.astype(np.int32)
        #add vertices to graph
        g.add_vertices(rows*cols+2)
        #get the neighbors for each pixel
        img0,img1,img2,img3,img4,img5,img6,img7=build_mats(img)

        #get B for N Link
        B=get_B(img,img0,img1,img2,img3,img4,img5,img6,img7)

        #calculate N link and K
        N_link(img,img0,img1,img2,img3,img4,img5,img6,img7,B)
        first_run=False
        img = img.astype(np.uint8)

        #build the t link edges
        fore_edges = t_link_edges(rows, cols, fore_node)
        back_edges = t_link_edges(rows, cols, back_node)
        
        #add eges to graph
        t_edges = np.vstack((fore_edges, back_edges))
        edges = np.vstack((n_edges,t_edges))
        g.add_edges(edges)
        
    Dm_back=bgGMM.score_samples(img.reshape(-1,c))
    Dm_fore=fgGMM.score_samples(img.reshape(-1,c))
    #modify scores
    Dm_back = -((Dm_back) + 1.5 * np.log(2 * np.pi))
    Dm_fore = -((Dm_fore) + 1.5 * np.log(2 * np.pi))
    #get the weights for t link
    t_weights=t_link(Dm_back,Dm_fore,mask, rows,cols)
    
    #change the weights for the edges in graph
    g.es['weight'] = np.hstack((n_weights,t_weights))
    
    #find the mincut for the graph
    cut = g.st_mincut(fore_node, back_node, capacity=g.es['weight'])
    energy=cut.value
    temp_set1, temp_set2 = cut.partition
    min_cut[0]= temp_set1 if fore_node in temp_set1 else temp_set2
    min_cut[1]= temp_set1 if back_node in temp_set1 else temp_set2 
    return min_cut, energy


def update_mask(mincut_sets, mask):
    # TODO: implement mask update step
    global last_run
    rows, cols = mask.shape

    #build idex matrix
    indexes = np.arange(rows*cols).reshape((rows,cols))
    
    #change mask
    if(not last_run):
        mask[(np.isin(indexes, list(mincut_sets[0]))) & (mask != GC_FGD) & (mask != GC_BGD)] = GC_PR_FGD
        mask[(np.isin(indexes, list(mincut_sets[1]))) & (mask != GC_BGD) & (mask != GC_FGD)] = GC_PR_BGD
    else:
        #in last run change the unknown pixels to known
        mask[(np.isin(indexes, list(mincut_sets[0]))) & (mask != GC_FGD) & (mask != GC_BGD)] = GC_FGD
        mask[(np.isin(indexes, list(mincut_sets[1]))) & (mask != GC_BGD) & (mask != GC_FGD)] = GC_BGD
    return mask

def check_convergence(energy):
    # TODO: implement convergence check
    global last_run, prev_energy
    
    #check if convergence
    if(abs(prev_energy - energy)<epsilon):
        last_run = True
        return False
    if(last_run):
        return True
    prev_energy = energy
    return False
    


def cal_metric(predicted_mask, gt_mask):
    # TODO: implement metric calculation
    print(gt_mask)
    
    #calculating accuracy
    similar = np.sum(predicted_mask == gt_mask)
    accuracy = similar/predicted_mask.size
    
    #calculating Jaccard similarity
    # Calculate intersection and union
    intersection = np.sum(predicted_mask & gt_mask)
    union = np.sum(predicted_mask | gt_mask)

    # Calculate Jaccard similarity
    if union == 0:
        jaccard = 1.0  # Both masks are empty, they are perfectly similar
    else:
        jaccard = intersection / union

    return accuracy,jaccard

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='teddy', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()



###########################################################################################################################
#                                                       Helper Functions                                                  #
###########################################################################################################################

'''takes list of means covarinces and weights for each foreground and background and it integrade them to the gmms'''
def build_GMM(bgGMM,fgGMM,back_means,back_covs,back_weigts,fore_means,fore_covs,fore_weigts):
    #background
    bgGMM.means_=np.array(back_means)
    bgGMM.covariances_=np.array(back_covs)
    bgGMM.weights_=np.array(back_weigts)
    bgGMM.precisions_ = np.linalg.inv(bgGMM.covariances_)
    bgGMM.precisions_cholesky_ = np.linalg.cholesky(bgGMM.precisions_)

    #foreground
    fgGMM.means_=np.array(fore_means)
    fgGMM.covariances_=np.array(fore_covs)
    fgGMM.weights_=np.array(fore_weigts)
    fgGMM.precisions_ = np.linalg.inv(fgGMM.covariances_)
    fgGMM.precisions_cholesky_ = np.linalg.cholesky(fgGMM.precisions_)
    return bgGMM, fgGMM

'''takes the orginal imgae and images of the 8 neighbors and it calculate B for N link equation'''
def get_B(img,img0,img1,img2,img3,img4,img5,img6,img7):
    stacked_neighbors = np.stack(np.array([img0,img1,img2,img3,img4,img5,img6,img7]))
    squared_diffs = (stacked_neighbors - img) ** 2
    squared_diffs_sum = np.sum(squared_diffs, axis=-1)
    expected_msd = np.mean(np.mean(squared_diffs_sum, axis=0))
    B = 1 / (2 * expected_msd)
    return B

'''takes image and return matrixes for each of the 8 neghbore where each elemnt is the neighbor in that size'''
def build_mats(img):
    rows,cols,c=img.shape
    img0,img1,img2,img3=np.zeros_like(img),np.zeros_like(img),np.zeros_like(img),np.zeros_like(img)
    img4,img5,img6,img7=np.zeros_like(img),np.zeros_like(img),np.zeros_like(img),np.zeros_like(img)
    img0[1:, :],img1[1:, 1:],img2[:, 1:],img3[:-1, 1:]=img[:-1, :],img[:-1, :-1],img[:, :-1],img[1:, :-1]
    img4[:-1, :],img5[:-1, :-1],img6[:, :-1],img7[1:, :-1]=img[1:, :],img[1:, 1:],img[:, 1:],img[:-1, 1:]
    return img0,img1,img2,img3,img4,img5,img6,img7

'''find the n link edges and weights plus calculate K for t link'''
def N_link(img,img0,img1,img2,img3,img4,img5,img6,img7,B):
    global K,n_edges,n_weights
    rows, cols, c = img.shape
    #matrxis of index in graph used to bulid links
    ind_img0=np.arange(rows*cols).reshape((rows, cols))[:-1, :].reshape(-1)
    ind_img1=np.arange(rows*cols).reshape((rows, cols))[:-1, :-1].reshape(-1)
    ind_img2=np.arange(rows*cols).reshape((rows, cols))[:, :-1].reshape(-1)
    ind_img3=np.arange(rows*cols).reshape((rows, cols))[1:, :-1].reshape(-1)
    ind_neg0=np.arange(rows*cols).reshape((rows, cols))[1:,:].reshape(-1)
    ind_neg1=np.arange(rows*cols).reshape((rows, cols))[1:, 1:].reshape(-1)
    ind_neg2=np.arange(rows*cols).reshape((rows, cols))[:, 1:].reshape(-1)
    ind_neg3=np.arange(rows*cols).reshape((rows, cols))[:-1, 1:].reshape(-1)

    #calculate weights
    stacked_neighbors = np.stack(np.array([img0,img1,img2,img3,img4,img5,img6,img7]))
    squared_diffs = np.power((stacked_neighbors - img),2)
    squared_diffs_sum = np.sum(squared_diffs, axis=-1)
    temp_weights=lama*50*np.exp(-1*B*squared_diffs_sum) # lama is used to get better results in complex imgaes 
    temp_weights[[1,3,5,7]]/=np.sqrt(2)
    weight0,weight1,weight2,weight3=temp_weights[0][1:,:].reshape(-1),temp_weights[1][1:, 1:].reshape(-1),temp_weights[2][:, 1:].reshape(-1),temp_weights[3][:-1, 1:].reshape(-1)
    link0=np.stack((ind_img0, ind_neg0), axis=-1)
    link1=np.stack((ind_img1, ind_neg1), axis=-1)
    link2=np.stack((ind_img2, ind_neg2), axis=-1)
    link3=np.stack((ind_img3, ind_neg3), axis=-1)

    #find K
    K=np.max(np.sum(temp_weights, axis =0))
    
    #add n edges and their weights
    n_edges=np.vstack((link0, link1,link2,link3))
    n_weights=np.hstack((weight0,weight1,weight2,weight3))

'''create the edges for t_link'''
def t_link_edges(n, m, const):
    array = np.arange(0, n*m)
    array_1 = np.full_like(array, const)
    tuple_array = np.column_stack((array, array_1))
    return tuple_array

'''find the weights for t links'''
def t_link(Dm_back,Dm_fore,mask, rows,cols):
        fore_weights = np.zeros((rows, cols)).reshape(-1)
        back_weights = np.zeros((rows, cols)).reshape(-1)
        mask_reshaped = mask.reshape(-1)
        
        #fill fore_weights and back_weights
        fore_weights = Dm_back
        back_weights = Dm_fore
        fore_weights[mask_reshaped==GC_FGD] = K
        fore_weights[mask_reshaped==GC_BGD] = 0
        back_weights[mask_reshaped == GC_BGD] = K
        back_weights[mask_reshaped == GC_FGD] = 0
        
        #merge weights
        weights = np.hstack((fore_weights,back_weights))
        
        return weights

'''Initialize K clusters for the foreground Gaussian Mixture Model (GMM) C1 RGB points, K number of clusters'''
def initialize_gmm_clusters(C1, K):
    # Calculate initial mean and covariance
    mu1 = np.mean(C1, axis=0)
    Sigma1 = np.cov(C1, rowvar=False)
    clusters = [(C1, mu1, Sigma1)]

    #loop and split
    for i in range(2, K+1):
        # Find the set Cn with the largest eigenvalue
        max_eigenvalue = -np.inf
        for j, (C, mu, Sigma) in enumerate(clusters):
            # Use OpenCV to compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = cv2.eigen(Sigma)[1:3]
            if np.max(eigenvalues) > max_eigenvalue:
                max_eigenvalue = np.max(eigenvalues)
                max_set_index = j
                max_eigenvector = eigenvectors[np.argmax(eigenvalues)]

        # Split the set Cn into two sets Ci and the remaining Cn
        Cn = clusters[max_set_index][0]
        en = max_eigenvector
        mun = clusters[max_set_index][1]
        Ci = Cn[np.all((en.T*Cn  <= en.T*mun).reshape((Cn.shape)), axis=1)]
        Cn_new = Cn[np.all((en.T*Cn  > en.T*mun).reshape((Cn.shape)), axis=1)]
        
        # Compute the new means and covariances
        if len(Ci) > 0:
            mu_i = np.mean(Ci, axis=0)
            Sigma_i = np.cov(Ci, rowvar=False)+ 1e-6 * np.eye(3)
        else:
            mu_i = np.zeros_like(mun)
            Sigma_i = np.zeros((len(mun), len(mun)))+ 1e-6 * np.eye(3)

        if len(Cn_new) > 0:
            mu_n = np.mean(Cn_new, axis=0)
            Sigma_n = np.cov(Cn_new, rowvar=False)+ 1e-6 * np.eye(3)
        else:
            mu_n = np.zeros_like(mun)
            Sigma_n = np.zeros((len(mun), len(mun)))+ 1e-6 * np.eye(3)

        # Update the clusters
        clusters[max_set_index] = (Cn_new, mu_n, Sigma_n)
        clusters.append((Ci, mu_i, Sigma_i))

    # Compute weights for each cluster
    total_points = np.sum([len(C) for C, _, _ in clusters])
    weights = [len(C) / total_points for C, _, _ in clusters]
    means=[M for _,M,_ in clusters]
    covs=[S for _,_,S in clusters]
    return means,covs, weights

###########################################################################################################################
#                                                end of Helper Functions                                                  #
###########################################################################################################################



if __name__ == '__main__':
    # Load an example image and define a bounding box around the object of interest
    args = parse()


    if args.input_img_path == '':
        input_path = f'data/imgs/{args.input_name}.jpg'
    else:
        input_path = args.input_img_path
    
    if args.use_file_rect:
        rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
    else:
        rect = tuple(map(int,args.rect.split(',')))


    img = cv2.imread(input_path)
    
    # Run the GrabCut algorithm on the image and bounding box
    mask, bgGMM, fgGMM = grabcut(img, rect)
    
    
    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]

    # Print metrics only if requested (valid only for course files)
    if args.eval:
        gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
        acc, jac = cal_metric(mask, gt_mask)
        print(f'Accuracy={acc}, Jaccard={jac}')

    # Apply the final mask to the input image and display the results
    img_cut = img * (mask[:, :, np.newaxis])
    cv2.imshow('Original Image', img)
    cv2.imshow('GrabCut Mask', 255 * mask)
    cv2.imshow('GrabCut Result', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
