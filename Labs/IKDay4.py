#IK Interdisciplinary College
#Linear Algebra
#Lecture 4 (only some activities)
#Instructor: Emily J. King

#######
# Generating noisy data around the line y=2x in R^2.
m=2 # slope of the line
sigma=0.1 # noise level
X=makepcadata(np.transpose(np.array([np.arange(1, m+1)])), 100, sigma)

# Plot data
plt.figure(1)
ax=plt.gca()
ax.grid('off')

plt.plot(X[:,0],X[:,1],'o')
plt.title("Along the line y="+str(m)+"x. Noise level="+str(sigma))

# Compute PCA
Xc=X-np.mean(X, axis=0) # zero-center columns
U, s, VT = np.linalg.svd(Xc, full_matrices=True) # compute SVD of centered-data
print("The singular values are "+str(s)) # show singular values
print("The principal components are the columns of")
print(np.transpose(VT)) # show principal components (columns of matrix)

S = np.zeros(X.shape)
np.fill_diagonal(S, s)
print("Double checking SVD is correct. This number should be very close to zero: "+str(np.linalg.norm(Xc-(U@S)@VT))) # double check USV^T = Xc

# Project onto PCA line of best fit
Ut=np.transpose(np.array([U[:,0]])) # truncate to first column
St=S[0,0] # truncate to upper left entry
Vt=np.array([VT[0]]) # truncate to first row of transposed matrix

Xt = (Ut*St)@Vt+np.mean(X, axis=0) # project down to span of first column of V and add back in the mean of the original vectors

plt.figure(2)
ax=plt.gca()
ax.grid('off')

plt.plot(Xt[:,0],Xt[:,1],'o')
plt.title("Projection to PCA line of best fit of data generated along line y="+str(m)+"x.\n Noise level="+str(sigma))

#plt.show() 

#################
# Generating noisy data around 2D subspace of R^4
sigma=0.1
W=np.array([[1, 1], [1, -1], [1, 1], [1, -1]]) # columns [1,1,1,1]^T, [1,-1,1,-1]^T span 2D subspace of R^4
X=makepcadata(W, 1000, sigma)

# Can't plot points in 4D

# Compute PCA
Xc=X-np.mean(X, axis=0) # zero-center columns
U, s, VT = np.linalg.svd(Xc, full_matrices=True) # compute SVD of centered-data
print("The singular values are "+str(s)) # show singular values
print("The principal components are the columns of")
print(np.transpose(VT)) # show principal components (columns of matrix)

S = np.zeros(X.shape)
np.fill_diagonal(S, s)
print("Double checking SVD is correct. This number should be very, very close to zero: "+str(np.linalg.norm(Xc-(U@S)@VT))) # double check USV^T = Xc

# Project onto coords of first two columns of V
Ut=U[:,0:2] # truncate to first two columns
St=S[0:2,0:2] # truncate to upper left 2x2 entries
Vt=VT[0:2] # truncate to first two rows of transposed matrix

print("This number in some sense measures the difference between the 2D subspace of R^4 that the data set points are near and the 2D subspace that is fit "+str(np.linalg.norm(np.transpose(Vt)@Vt-W@np.transpose(W)/4)))

Xt = Ut@St # project down to coords of first two columns of V.  this is called the "scores"

plt.figure(3)
ax=plt.gca()
ax.grid('off')

plt.plot(Xt[:,0],Xt[:,1],'o')
plt.title("Projection to coords of first two columns of V. Noise level="+str(sigma))

#plt.show() 

###############
# Generating noisy data around random 3D subspace of R^50
sigma=0.1
W=np.linalg.qr(np.random.randn(50,3))[0] # generate basis of random 3D subspace of R^50
X=makepcadata(W, 1000, sigma)

# Can't plot points in 50D

# Compute PCA
Xc=X-np.mean(X, axis=0) # zero-center columns
U, s, VT = np.linalg.svd(Xc, full_matrices=True) # compute SVD of centered-data
#print("The singular values are "+str(s)) # show singular values

S = np.zeros(X.shape)
np.fill_diagonal(S, s)
print("Double checking SVD is correct. This number should be very, very close to zero: "+str(np.linalg.norm(Xc-(U@S)@VT))) # double check USV^T = Xc

# Plot scree plot
plt.figure(4)
ax=plt.gca()
ax.grid('off')

plt.plot(s)
plt.title("Scree plot of singular values of points near random 3D subspace of R^50.\n Noise level="+str(sigma))

# Compare fit subspace to subspace used to generate the data points
#Ut=U[:,0:3] # truncate to first three columns
#St=S[0:3,0:3] # truncate to upper left 3x3 entries
Vt=VT[0:3] # truncate to first three rows of transposed matrix

print("This number in some sense measures the difference between the 3D subspace of R^50 that the data set points are near and the 3D subspace that is fit "+str(np.linalg.norm(np.transpose(Vt)@Vt-W@np.transpose(W))))


plt.show() 

