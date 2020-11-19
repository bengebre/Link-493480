### SkyDelta ###
#a toolbox for Solar System moving object detection
#this is a stripped down version that just does linking

import numpy as np

#calcs velocity vector of a blob with respect to all others
#then for each 'all other' within pixel per day (PPD) constraints 
#checks for a cluster of similar velocity vectors
def link_blobs(bbf,times,pct,ppd=[1,100],min_frames=3,max_span=150):
	from sklearn.neighbors import KDTree
	from numpy import linalg as LA

	links = []
	j = 0
	for b in bbf:
		sub_idx = np.where((bbf[:,3] != b[3]) & (LA.norm(b[0:2]-bbf[:,0:2],axis=1) <= max_span)) #span by pixels
		dxdt = blob_dxdt(b,bbf[sub_idx],times)
		if len(dxdt)==0:
			j+=1
			continue
		l_dxdt = LA.norm(dxdt,axis=1)
		thresh = np.logical_and(l_dxdt < ppd[1],l_dxdt > ppd[0]) #pix/day
		tree = KDTree(dxdt)
		for i in np.arange(0,len(dxdt)):
			if thresh[i]:
				ind = tree.query_radius(np.asarray([dxdt[i]]), r=pct*l_dxdt[i])
				bbf_idx = np.append(j,sub_idx[0][ind[0]])
				if (len(np.unique(bbf[bbf_idx,3])) >= min_frames):
					l = list(np.sort(bbf_idx))
					links.append(l)
            
		j+=1
        
	#uniq list for now (though there is count information in the non-uniqued list)
	links = [list(l) for l in set(tuple(l) for l in links)] #uniq list of list
	return links

#pixel velocity of one blob with respect to other blobs
def blob_dxdt(origin,bbf,times):
	if len(np.shape(bbf))==1:
		bbf = np.asarray([bbf])
	origin_time = times[origin[3].astype(int)]
	bbf_dt = times[bbf[:,3].astype(int)] - origin_time
	bbf_dx = bbf[:,0:2] - origin[0:2]

	return bbf_dx/np.transpose(np.vstack((bbf_dt,bbf_dt)))

#returns [y,x,obj_id,frame_num]
def blobs_by_frame(frames,centroid=True):
	start_obj_id = 0
	for i,f in enumerate(frames):
		if i == 0:
			bbf = blob_coords(f,idx0=start_obj_id,fnum=i,centroid=centroid)
		else:
			fb = blob_coords(f,idx0=start_obj_id,fnum=i,centroid=centroid)
			bbf = np.vstack((bbf,fb))

		start_obj_id = np.max(np.vstack(bbf)[:,2]) + 1

	return bbf

#returns the centroids of threshed blobs [y,x,obj_id,frame_num]
def blob_coords(frame,idx0=0,fnum=None,centroid=True):
	from numpy import linalg as LA
	from skimage import measure

	labeled = measure.label(frame,connectivity=2)
	blobs = measure.regionprops(labeled)
	bcoords = []
	bidx = []
	i = idx0
	for b in blobs:

		if centroid: #centroid of joined pixels (less blobs)
			b_centroid = np.asarray([b.centroid]) #centroid only
			bcoords.append(b_centroid) 
			bidx.append(np.ones((np.shape(b_centroid)[0],1))*i)

		else: #all threshed pixels (more blobs)
			bcoords.append(b.coords)
			bidx.append(np.ones((np.shape(b.coords)[0],1))*i)

		i+=1
            
	if len(bcoords) == 0:
		return []
    
	a = np.concatenate(bcoords)
	b = np.concatenate(bidx)
    
	if fnum != None:
		c = np.ones((len(b),1))*fnum
		frame_blobs = np.hstack((a,b,c))
	else:
		frame_blobs = np.hstack((a,b))

	return frame_blobs
