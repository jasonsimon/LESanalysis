#!/usr/bin/python3

import datetime
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
import sklearn.model_selection
import sklearn.metrics
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pylab as plt
import matplotlib.colors
import cmocean as cmo
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
from matplotlib import rcParams
rcParams['text.usetex'] = True
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'cm'
rcParams['xtick.major.size'] = 2.0
rcParams['ytick.major.size'] = 2.0
rcParams['legend.handletextpad'] = 0
rcParams['legend.handlelength'] = 0
rcParams['legend.borderaxespad'] = 0.2
rcParams['legend.labelspacing'] = 0.20
rcParams['legend.framealpha'] = 0.00
rcParams['legend.borderpad'] = 0.1
rcParams['legend.handleheight'] = 1.0
rcParams['axes.linewidth'] = 0.5



def pca():

	#Read in surface stats
	db = pickle.load(open('workspace/SFstats.pck','rb'))
	metrics, les_db = readmetrics(db)

	var = 'LWP'
	metric = metrics[var]
	#m2 = m #& (metric < 7000)
	m2 = metric != -9999
	print(m2)
	print([date for date in les_db])
	sh_sd = ( db['sh_var'][m2] )**0.5
	sh_max = ( db['sh_max'][m2] )
	lh_sd = ( db['lh_var'][m2] )**0.5
	lh_max = ( db['lh_max'][m2] )
	lw_sd = ( db['lw_var'][m2] )**0.5
	lw_max = ( db['lw_max'][m2] )
	L0 =  db['sh_L0_0.75'][m2]
	L90 =  db['sh_L90_0.75'][m2]
	qv_ls = ( db['qv_ls_4km'][:] )
	th_ls = ( db['th_ls_4km'][:] )

	X = np.array([L0, L90, sh_max, lh_max, lw_max, sh_sd, lh_sd, lw_sd, qv_ls, th_ls]).T
	y = metric[m2]

	print(X.shape)
	print(y.shape)


	pca = PCA(n_components=X.shape[1]).fit(X)

	plt.figure(figsize=(10,5))
	plt.scatter(X[:, 0], X[:, 1], alpha=.3, label='samples')
	for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
		comp = comp * var  # scale component by its variance explanation power
		plt.plot([0, comp[0]], [0, comp[1]], label=f"Component {i}", linewidth=1,color=f"C{i + 2}")
		#plt.gca().set(aspect='equal', title="2-dimensional dataset with principal components", xlabel='first feature', ylabel='second feature')
		plt.gca().set(title="2-dimensional dataset with principal components", xlabel='first feature', ylabel='second feature')
	plt.legend()
	plt.tight_layout()
	plt.show()

def readmetrics(db):

	#Read in LES output
	file = 'workspace/LESoutput.pck'
	les_db = pickle.load(open(file,'rb'))

	#Assemble metrics
	metrics = {}
	for var in ['TKE','LWP']:
		metrics[var] = []
		#Perform analysis at each date
		for date in les_db:
			if (np.sum(les_db[date]['HMG'][var] != -9999) >= 90) & (np.sum(les_db[date]['HTG'][var] != -9999) >= 90):
				#tmp = ( les_db[date]['HTG'][var]-les_db[date]['HMG'][var] ) #/ np.mean(les_db[date]['HTG'][var])
				#tmp = ( les_db[date]['HTG'][var]/les_db[date]['HMG'][var] )
				eps = 1e-12
				tmp = np.log2( (les_db[date]['HTG'][var]+eps)/(les_db[date]['HMG'][var]+eps) )
				#tmp = les_db[date]['HMG'][var]
				tmp = np.mean(tmp)
				if tmp < 0:
					print(var," < 0")
					print(tmp)
					print(date)
					print(les_db[date]['HTG'][var])
				metrics[var].append( tmp )
				#metrics[var].append(np.sum(les_db[date]['HTG'][var])-np.sum(les_db[date]['HMG'][var]))
			else:
				metrics[var].append(-9999)
		metrics[var] = np.array(metrics[var])
	#m = metrics['LWP'] != -9999
	#print(m)

	return metrics, les_db


def linear_reg(var, var1, aa):

	#Read in surface stats
	db = pickle.load(open('workspace/SFstats.pck','rb'))
	metrics, les_db = readmetrics(db)

	metric = metrics[var]
	tmp = ( db['th_ls_4km'][:] )
	tmplh = db['lh_mean'][:]
	tmpsh = db['sh_mean'][:]
	#m2 = m #& (metric < 7000)
	m2 = metric != -9999
	#m2 = metric < 6000
	#m2 = tmp <= 0.0
	#m2 = tmplh < tmpsh


	print(m2)
	tmp = ([date for date in les_db])
	print(tmp)



	sh_sd = ( db['sh_var'][m2] )**0.5
	sh_var = ( db['sh_var'][m2] )
	sh_mean = ( db['sh_mean'][m2] )
	sh_max = ( db['sh_max'][m2] )
	lh_sd = ( db['lh_var'][m2] )**0.5
	lh_var = ( db['lh_var'][m2] )
	lh_mean = ( db['lh_mean'][m2] )
	lh_max = ( db['lh_max'][m2] )
	lw_sd = ( db['lw_var'][m2] )**0.5
	lw_var = ( db['lw_var'][m2] )
	lw_mean = ( db['lw_mean'][m2] )
	lw_max = ( db['lw_max'][m2] )
	bo_sd = ( db['bo_var'][m2] )**0.5
	bo_var = ( db['bo_var'][m2] )
	bo_mean = ( db['bo_mean'][m2] )
	bo_max = ( db['bo_max'][m2] )
	ef_sd = ( db['ef_var'][m2] )**0.5
	ef_var = ( db['ef_var'][m2] )
	ef_mean = ( db['ef_mean'][m2] )
	ef_max = ( db['ef_max'][m2] )
	L0 =  db['sh_L0_0.75'][m2]
	L90 =  db['sh_L90_0.75'][m2]
	ws =  db['ws'][m2]
	qv250m = ( db['qv_ls_250m'][m2] )
	qv500m = ( db['qv_ls_500m'][m2] )
	qv1km = ( db['qv_ls_1km'][m2] )
	qv2km = ( db['qv_ls_2km'][m2] )
	qv3km = ( db['qv_ls_3km'][m2] )
	qv4km = ( db['qv_ls_4km'][m2] )
	th250m = ( db['th_ls_250m'][m2] )
	th500m = ( db['th_ls_500m'][m2] )
	th1km = ( db['th_ls_1km'][m2] )
	th2km = ( db['th_ls_2km'][m2] )
	th3km = ( db['th_ls_3km'][m2] )
	th4km = ( db['th_ls_4km'][m2] )
	w250m = ( db['w_ls_250m'][m2] )
	w500m = ( db['w_ls_500m'][m2] )
	w1km = ( db['w_ls_1km'][m2] )
	w2km = ( db['w_ls_2km'][m2] )
	w3km = ( db['w_ls_3km'][m2] )
	w4km = ( db['w_ls_4km'][m2] )

	#X = np.array([db['SH_L0_0.75'][m2],db['SH_L90'][m2],db['SH_var'][m2],db['SH_skew'][m2],db['SH_kurt'][m2]]).T
	#X = np.array([db['SH_L0_0.50'][m2],db['SH_L90_0.50'][m2],db['SH_var'][m2],db['SH_skew'][m2],db['SH_kurt'][m2]]).T
	#X = np.array([db['sh_L0_0.75'][m2],db['sh_L90_0.75'][m2],
	#              db['lw_L0_0.75'][m2],db['lw_L90_0.75'][m2],
	#              db['lh_L0_0.75'][m2],db['lh_L90_0.75'][m2],
	#              db['sh_var'][m2],db['lh_var'][m2],db['lw_var'][m2],
	#              db['sh_skew'][m2],db['lh_skew'][m2],db['lw_skew'][m2],
	#              db['sh_kurt'][m2],db['lh_kurt'][m2],db['lw_kurt'][m2],
	#              db['ws'][m2],db['theta'][m2]]).T
	#X = np.array([db['sh_L0_0.75'][m2],db['sh_L90_0.75'][m2],db['sh_var'][m2]]).T
	#X = np.array([db['sh_L0_0.75'][m2],db['sh_L0_0.50'][m2],db['sh_L0_0.25'][m2],db['sh_L90_0.75'][m2],db['sh_L90_0.50'][m2],db['sh_L90_0.25'][m2]]).T
	#X = np.array([L0, L90, sh_max, lh_max, lw_max, sh_sd, lh_sd, lw_sd, qv_ls, th_ls, ws]).T
	#X = np.array([L0, lw_sd]).T

	#Xlist = [L0, th4km, L90, sh_max, lh_max, lw_max, sh_sd, lh_sd, lw_sd, qv4km, ws, sh_mean, lh_mean, lw_mean, bo_mean, bo_sd, ef_mean, ef_sd]
	#Xnames = ['L0', 'th4km', 'L90', 'sh_max', 'lh_max', 'lw_max', 'sh_sd', 'lh_sd', 'lw_sd', 'qv4km', 'ws', 'sh_mean', 'lh_mean', 'lw_mean', 'bo_mean', 'bo_sd', 'ef_mean', 'ef_sd']
	#Xlist = [L0, L90, sh_sd, lh_sd, lw_sd, ws, sh_mean, lh_mean, lw_mean, bo_mean, bo_sd, ef_mean, ef_sd]
	#Xnames = ['L0', 'L90', 'sh_sd', 'lh_sd', 'lw_sd', 'ws', 'sh_mean', 'lh_mean', 'lw_mean', 'bo_mean', 'bo_sd', 'ef_mean', 'ef_sd']

	#Xlist = [L0, L90, sh_var]
	#Xnames = ['L0,L90,sh_var','L0+L90+sh_var','L0+L90+sh_var']

	if var1 == "l0":
		Xlist = [L0]#, ef_mean, ef_var]
		Xnames = ['L0']#,ef_mean,ef_var','L0+L90+sh_var','L0+L90+sh_var']
	elif var1 == "sh":
		Xlist = [sh_mean, sh_sd, "both"]#, lh_mean, lh_var, ef_mean, ef_var]
		Xnames = ['SH mean', 'SH sd', 'SH mean, SH sd']
		Xunits = ['(W m$^2$)','(W m$^2$)','(W m$^2$)','(W m$^2$)','(W m$^2$)','(W m$^2$)','(-)','(-)']
	elif var1 == "lh":
		Xlist = [lh_mean, lh_sd, "both"]#, lh_mean, lh_var, ef_mean, ef_var]
		Xnames = ['LH mean', 'LH sd', 'LH mean, LH sd']
		Xunits = ['(W m$^2$)','(W m$^2$)','(W m$^2$)','(W m$^2$)','(W m$^2$)','(W m$^2$)','(-)','(-)']
	elif var1 == "lw":
		Xlist = [lw_mean, lw_sd, "both"]#, lh_mean, lh_var, ef_mean, ef_var]
		Xnames = ['LW mean', 'LW sd', 'LW mean, LW sd']
		Xunits = ['(W m$^2$)','(W m$^2$)','(W m$^2$)','(W m$^2$)','(W m$^2$)','(W m$^2$)','(-)','(-)']
	elif var1 == "ef":
		Xlist = [ef_mean, ef_sd, "both"]#, lh_mean, lh_var, ef_mean, ef_var]
		Xnames = ['EF mean', 'EF sd', 'EF mean, EF sd']
	else:
		Xlist = [L0, qv250m, qv500m, qv1km, qv2km, qv3km, qv4km, th250m, th500m, th1km, th2km, th3km, th4km]
		Xnames = ['L0', 'qv250m', 'qv500m', 'qv1km', 'qv2km', 'qv3km', 'qv4km','th250m', 'th500m', 'th1km', 'th2km', 'th3km', 'th4km']

	#Xlist = [th_ls, L90, sh_max, lh_max, lw_max, sh_sd, lh_sd, lw_sd, qv_ls, ws]
	#Xnames = ['th_ls', 'L90', 'sh_max', 'lh_max', 'lw_max', 'sh_sd', 'lh_sd', 'lw_sd', 'qv_ls', 'ws']
	#X = np.array([L0]).T
	#X = np.array([L0, L90, sh_sd]).T
	#X = np.array([L0, sh_sd]).T
	#X = np.array([L0, qv_ls]).T
	#X = np.array([L0, lw_sd]).T
	y = metric[m2]
	#y /= (sh_mean)
	print(len(m2))
	print(len(y))
	print(y)


	if aa < 2:
		looprange = range(len(Xlist))
		#plt.figure(figsize=(15,5))
		plt.figure(figsize=(min(15,3*len(Xlist)),5))
	else:
		looprange = [0]
		plt.figure(figsize=(5,5))

	nr = 2
	#nc = int( np.ceil( float(X.shape[1]) / nr ) + 1 )
	nc = len(looprange)
	pn = 0

	for i in looprange:

		if not aa:
			if Xlist[i] == "both":
				X = np.array( [ np.copy(Xlist[0]), np.copy(Xlist[1]) ] ).T
			else:
				X = np.array(np.copy(Xlist[i]))
				X = X.reshape(-1,1)
		elif aa == 1:
			Xtmp = [L0]
			if i:
				Xtmp.append(np.copy(Xlist[i]))
			X = np.array(Xtmp).T
		else:
			X = np.array(Xlist).T

		print(X.shape)
		print(y.shape)

		#regr = RandomForestRegressor(random_state=0,n_estimators=250,oob_score=True).fit(X,y)
		regr = LinearRegression().fit(X, y)
		ypred = regr.predict(X)
		print(sklearn.metrics.explained_variance_score(y,ypred))


		clf = LinearRegression()
		#clf = RandomForestRegressor(n_estimators=250)
		scores = sklearn.model_selection.cross_val_score(clf,X,y,cv=2,scoring='explained_variance')
		print(np.mean(scores))
		cv0 = np.mean(scores)
		scores = sklearn.model_selection.cross_val_score(clf,X,y,cv=2,scoring='r2')
		#scores = sklearn.model_selection.cross_val_score(clf,X,y,cv=2,scoring='neg_mean_squared_error')
		print(np.mean(scores))
		cv1 = np.mean(scores)



		pn += 1
		p = plt.subplot(nr,nc,pn)
		if not aa:
			if Xlist[i] == "both":
				s = 100.0*y#/np.max(y)
				plt.scatter(X[:,0],X[:,1],s=s,marker='o',color='b',alpha=0.6,edgecolors='k')
				plt.xlabel(Xnames[0] + " " + Xunits[0])#,fontsize=30)
				plt.ylabel(Xnames[1] + " " + Xunits[1])#,fontsize=30)
			else:
				#plt.plot(X,y,'bo')
				plt.scatter(X,y,marker='o',color='b',alpha=0.6,edgecolors='k')
		else:
			plt.plot(X[:,np.min([i,1])],y,'bo')
		#p.set_yscale("log")
		#p.set_xscale("log")
		#plt.xlabel('Predictor %i' % i)#,fontsize=30)
		if not Xlist[i] == "both":
			if aa == 2:
				plt.xlabel('L0 (km)')#,fontsize=30)
			else:
				#plt.xlabel(Xnames[i].replace("_","\_"))#,fontsize=30)
				plt.xlabel(Xnames[i] + " " + Xunits[i])#,fontsize=30)
			if not i: 
				#plt.ylabel('$\overline{\Delta %s}$ (HTG - HMG)' % var)#,fontsize=30)
				plt.ylabel(r'$\overline{\log_2\left(\frac{%s_{\mbox{\tiny HTG}}}{%s_{\mbox{\tiny HMG}}}\right)}$' % (var,var))#,fontsize=30)
			else:
				plt.yticks(visible=False)
		#plt.xticks(fontsize=20)
		#plt.yticks(fontsize=20)
		#plt.title(r'$\overline{%s_{HTG} - %s_{HMG}}$ ($R^2 = %.2f$)' % (var,var,sklearn.metrics.explained_variance_score(y,ypred)))#,fontsize=25)


		modelr2 = sklearn.metrics.explained_variance_score(y,ypred)
		label = '$R^2=%0.2f$\n$EV_{\mbox{cv}}=%0.2f$\n$R^2_{\mbox{cv}}=%0.2f$' % (modelr2,cv0,cv1)
		#label=r"$R^2=%0.2f$" % modelr2
		#pn += 1
		p = plt.subplot(nr,nc,pn+nc)
		#p = plt.subplot(1,1,1,aspect='equal')
		plt.plot(y,y,'tab:red',label=label,zorder=1)
		plt.scatter(ypred,y,marker='o',color='b',alpha=0.6,edgecolors='k',zorder=2)
		#leg = plt.legend(loc='upper left')
		leg = plt.legend(loc='best')
		#leg.get_frame().set_alpha(0.0)
		leg.get_frame().set_linewidth(0.0)

		#p.set_yscale("log")
		#p.set_xscale("log")
		#plt.xlabel('Original')#,fontsize=30)
		#plt.xlabel('Predictor %i' % i)#,fontsize=30)
		if aa == 2:
			plt.xlabel(r'Model(%s)' % Xnames[i].replace("_","\_"))#,fontsize=30)
		elif not aa or not i:
			#plt.xlabel(r'%s' % Xnames[i].replace("_","\_"))#,fontsize=30)
			plt.xlabel(r'Linear(%s)' % Xnames[i].replace("_","\_"))#,fontsize=30)
		else:
			plt.xlabel(r'L0,%s' % Xnames[i].replace("_","\_"))#,fontsize=30)

		if not i: 
			#plt.ylabel('Predicted')#,fontsize=30)
			plt.ylabel('WRF')#,fontsize=30)
		else:
			plt.yticks(visible=False)
		#plt.xticks(fontsize=20)
		#plt.yticks(fontsize=20)
		#plt.title(r'$\overline{%s_{HTG} - %s_{HMG}}$ ($R^2 = %.2f$)' % (var,var,sklearn.metrics.explained_variance_score(y,ypred)),fontsize=12)
		"""
		if aa == 2:
			plt.title(r'$R^2=%0.2f$, $EV_{\mbox{cv}}=%0.2f$, $R^2_{\mbox{cv}}=%0.2f$' % (sklearn.metrics.explained_variance_score(y,ypred),cv0,cv1),fontsize=10)
		else:
			#plt.title(r'%0.2f, %0.2f, %0.2f' % (sklearn.metrics.explained_variance_score(y,ypred),cv0,cv1),fontsize=10)
			pass
		"""

	if aa == 2:
		tmp0 = "single"
		tmp1 = "l0"
	else:
		tmp0 = "triple"
		tmp1 = "l0%smv" % var1

	filename = "%s_%s_%s.pdf" % (tmp0, tmp1, var.lower())
	print(filename)

	plt.tight_layout()
	plt.show()
	#plt.savefig(filename)



if __name__ == "__main__":
	linear_reg("TKE","sh",0)
