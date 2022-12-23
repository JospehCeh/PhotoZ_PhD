import os,sys
import numpy as np
from astropy.io import fits
#test



#####################################################
#####################################################
#####################################################
#raw spectra from averaged spectra:

l0=['SPEC2','SPEC3','SPEC9','SPEC13','SPEC19','SPEC24','SPEC25','SPEC30','SPEC31','SPEC32','SPEC33','SPEC34','SPEC35','SPEC36','SPEC37','SPEC45','SPEC47','SPEC49','SPEC51','SPEC55','SPEC57','SPEC58','SPEC59','SPEC61','SPEC62','SPEC66','SPEC67','SPEC68','SPEC69','SPEC70','SPEC71','SPEC72','SPEC73','SPEC77','SPEC79','SPEC80','SPEC83','SPEC84','SPEC85','SPEC86','SPEC87','SPEC89','SPEC91','SPEC93','SPEC96','SPEC97']

l1=['SPEC102','SPEC106','SPEC107','SPEC109','SPEC110','SPEC111','SPEC112','SPEC113','SPEC114','SPEC115','SPEC117','SPEC118','SPEC120','SPEC121','SPEC123','SPEC127','SPEC128','SPEC132','SPEC134','SPEC135','SPEC137','SPEC138','SPEC141','SPEC149','SPEC151','SPEC152','SPEC156','SPEC160','SPEC161','SPEC164','SPEC171','SPEC178','SPEC179','SPEC181','SPEC182','SPEC184','SPEC185','SPEC186','SPEC187','SPEC188','SPEC189','SPEC191','SPEC192','SPEC193','SPEC194','SPEC196','SPEC197','SPEC198']

l2=['SPEC204','SPEC205','SPEC210','SPEC214','SPEC218','SPEC221','SPEC222','SPEC223','SPEC226','SPEC227','SPEC231','SPEC233','SPEC234','SPEC235','SPEC236','SPEC237','SPEC238','SPEC240','SPEC242','SPEC243','SPEC244','SPEC245','SPEC246','SPEC248','SPEC249','SPEC250','SPEC252','SPEC253','SPEC258','SPEC259','SPEC260','SPEC261','SPEC262','SPEC264','SPEC265','SPEC266','SPEC267','SPEC268','SPEC271','SPEC274','SPEC275','SPEC276','SPEC277','SPEC278','SPEC279','SPEC280','SPEC281','SPEC282','SPEC283','SPEC287','SPEC288','SPEC291','SPEC292','SPEC294','SPEC295','SPEC296','SPEC297','SPEC298']

l3=['SPEC301','SPEC302','SPEC303','SPEC305','SPEC306','SPEC307','SPEC308','SPEC309','SPEC313','SPEC315','SPEC317','SPEC318','SPEC319','SPEC321','SPEC322','SPEC323','SPEC324','SPEC325','SPEC326','SPEC327','SPEC328','SPEC329','SPEC331','SPEC332','SPEC333','SPEC334','SPEC335','SPEC336','SPEC337','SPEC338','SPEC339','SPEC340','SPEC341','SPEC343','SPEC344','SPEC345','SPEC348','SPEC349','SPEC350','SPEC351','SPEC352','SPEC353','SPEC354','SPEC355','SPEC357','SPEC358','SPEC359','SPEC360','SPEC361','SPEC362','SPEC363','SPEC364','SPEC365','SPEC366','SPEC367','SPEC368','SPEC369','SPEC370','SPEC371','SPEC372','SPEC373','SPEC374','SPEC375','SPEC376','SPEC377','SPEC378','SPEC379','SPEC380','SPEC381','SPEC382','SPEC383','SPEC384','SPEC385','SPEC386','SPEC387','SPEC388','SPEC389','SPEC390','SPEC391','SPEC392','SPEC393','SPEC394','SPEC395','SPEC396','SPEC397','SPEC398','SPEC399']

l4=['SPEC400','SPEC401','SPEC402','SPEC403','SPEC404','SPEC405','SPEC406','SPEC407','SPEC408','SPEC409','SPEC410','SPEC411','SPEC412','SPEC413','SPEC414','SPEC415','SPEC416','SPEC417','SPEC418','SPEC419','SPEC420','SPEC421','SPEC422','SPEC423','SPEC424','SPEC425','SPEC426','SPEC427','SPEC428','SPEC429','SPEC430','SPEC431','SPEC434','SPEC435','SPEC436','SPEC437','SPEC438','SPEC439','SPEC440','SPEC441','SPEC442','SPEC443','SPEC444','SPEC445','SPEC446','SPEC447','SPEC448','SPEC449','SPEC450','SPEC451','SPEC452','SPEC453','SPEC454','SPEC455','SPEC456','SPEC457','SPEC458','SPEC459','SPEC460','SPEC461','SPEC462','SPEC463','SPEC464','SPEC465','SPEC466','SPEC467','SPEC468','SPEC469','SPEC470','SPEC471','SPEC472','SPEC474','SPEC475','SPEC477','SPEC478','SPEC479','SPEC480','SPEC481','SPEC482','SPEC483','SPEC488','SPEC490','SPEC492','SPEC493','SPEC494','SPEC496','SPEC497','SPEC499']

l5=['SPEC500','SPEC501','SPEC503','SPEC504','SPEC505','SPEC506','SPEC507','SPEC508','SPEC509','SPEC510','SPEC511','SPEC512','SPEC513','SPEC516','SPEC517','SPEC519','SPEC520','SPEC523','SPEC524','SPEC525','SPEC526','SPEC527','SPEC528','SPEC529','SPEC530','SPEC531','SPEC532','SPEC533','SPEC535','SPEC536','SPEC537','SPEC539','SPEC540','SPEC541','SPEC542','SPEC543','SPEC544','SPEC545','SPEC546','SPEC547','SPEC548','SPEC549','SPEC550','SPEC551','SPEC552','SPEC553','SPEC554','SPEC556','SPEC557','SPEC558','SPEC559','SPEC560','SPEC562','SPEC563','SPEC564','SPEC565','SPEC566','SPEC567','SPEC568','SPEC569','SPEC570','SPEC571','SPEC572','SPEC573','SPEC574','SPEC575','SPEC576','SPEC577','SPEC578','SPEC579','SPEC580','SPEC582','SPEC583','SPEC584','SPEC585','SPEC586','SPEC587','SPEC588','SPEC589','SPEC590','SPEC593','SPEC594','SPEC595','SPEC596','SPEC597','SPEC598','SPEC599']

l6=['SPEC600','SPEC601','SPEC602','SPEC603','SPEC604','SPEC605','SPEC606','SPEC608','SPEC609','SPEC610','SPEC611','SPEC612','SPEC613','SPEC617','SPEC618','SPEC620','SPEC621','SPEC622','SPEC623','SPEC624','SPEC625','SPEC626','SPEC627','SPEC628','SPEC629','SPEC630','SPEC631','SPEC633','SPEC634','SPEC635','SPEC636','SPEC637','SPEC638','SPEC639','SPEC640','SPEC641','SPEC642','SPEC643','SPEC644','SPEC645','SPEC646','SPEC647','SPEC648','SPEC649','SPEC650','SPEC651','SPEC652','SPEC653','SPEC654','SPEC655','SPEC656','SPEC657','SPEC658','SPEC660','SPEC661','SPEC662','SPEC663','SPEC664','SPEC667','SPEC668','SPEC669','SPEC670','SPEC671','SPEC672','SPEC673','SPEC674','SPEC675','SPEC676','SPEC677','SPEC678','SPEC679','SPEC680','SPEC681','SPEC682','SPEC683','SPEC684','SPEC685','SPEC686','SPEC687','SPEC689','SPEC690','SPEC691','SPEC692','SPEC693','SPEC694','SPEC695','SPEC696','SPEC697','SPEC698']


l7=['SPEC700','SPEC701','SPEC702','SPEC703','SPEC704','SPEC705','SPEC706','SPEC707','SPEC708','SPEC710','SPEC711','SPEC713','SPEC714','SPEC715','SPEC716','SPEC717','SPEC718','SPEC719','SPEC720','SPEC721','SPEC722','SPEC723','SPEC724','SPEC725','SPEC726','SPEC727','SPEC728','SPEC729','SPEC730','SPEC731','SPEC732','SPEC733','SPEC734','SPEC735','SPEC736','SPEC737','SPEC738']




##########################################################################

sel_0 = ['SPEC91', 'SPEC79', 'SPEC9', 'SPEC3', 'SPEC61', 'SPEC62', 'SPEC67', 'SPEC69', 'SPEC68', 'SPEC25', 'SPEC87', 'SPEC86', 'SPEC85', 'SPEC84', 'SPEC83', 'SPEC80', 'SPEC72', 'SPEC70', 'SPEC58', 'SPEC59', 'SPEC55', 'SPEC57', 'SPEC36', 'SPEC37', 'SPEC34', 'SPEC32', 'SPEC33', 'SPEC30', 'SPEC31'] 
sel_1 =  ['SPEC132', 'SPEC152', 'SPEC137', 'SPEC111', 'SPEC115', 'SPEC192', 'SPEC193', 'SPEC191', 'SPEC178', 'SPEC197', 'SPEC194', 'SPEC151', 'SPEC123', 'SPEC109', 'SPEC107', 'SPEC128', 'SPEC102', 'SPEC184', 'SPEC186', 'SPEC181', 'SPEC182', 'SPEC141', 'SPEC161', 'SPEC160'] 
sel_2 = ['SPEC271', 'SPEC277', 'SPEC278', 'SPEC253', 'SPEC252', 'SPEC218', 'SPEC238', 'SPEC210', 'SPEC235', 'SPEC234', 'SPEC233', 'SPEC214', 'SPEC231', 'SPEC291', 'SPEC292', 'SPEC294', 'SPEC297', 'SPEC296', 'SPEC260', 'SPEC248', 'SPEC205', 'SPEC264', 'SPEC265', 'SPEC267', 'SPEC242', 'SPEC243', 'SPEC246', 'SPEC244', 'SPEC245', 'SPEC227', 'SPEC221', 'SPEC222', 'SPEC223', 'SPEC282', 'SPEC280', 'SPEC281'] 
sel_3 =  ['SPEC351', 'SPEC379', 'SPEC372', 'SPEC376', 'SPEC375', 'SPEC315', 'SPEC394', 'SPEC396', 'SPEC391', 'SPEC393', 'SPEC398', 'SPEC343', 'SPEC344', 'SPEC364', 'SPEC348', 'SPEC363', 'SPEC321', 'SPEC306', 'SPEC325', 'SPEC301', 'SPEC387', 'SPEC386', 'SPEC385', 'SPEC384', 'SPEC383', 'SPEC382', 'SPEC381', 'SPEC380', 'SPEC388'] 
sel_4 =  ['SPEC471', 'SPEC447', 'SPEC441', 'SPEC442', 'SPEC488', 'SPEC480', 'SPEC461', 'SPEC412', 'SPEC410', 'SPEC438', 'SPEC415', 'SPEC414', 'SPEC419', 'SPEC436', 'SPEC478', 'SPEC457', 'SPEC475', 'SPEC455', 'SPEC454', 'SPEC453', 'SPEC452', 'SPEC474', 'SPEC499', 'SPEC493', 'SPEC496', 'SPEC494', 'SPEC418', 'SPEC466', 'SPEC468', 'SPEC402', 'SPEC463', 'SPEC460', 'SPEC407', 'SPEC409', 'SPEC421', 'SPEC424', 'SPEC469'] 
sel_5 = ['SPEC509', 'SPEC501', 'SPEC500', 'SPEC525', 'SPEC524', 'SPEC505', 'SPEC504', 'SPEC506', 'SPEC580', 'SPEC583', 'SPEC585', 'SPEC584', 'SPEC520', 'SPEC570', 'SPEC517', 'SPEC527', 'SPEC516', 'SPEC571', 'SPEC572', 'SPEC573', 'SPEC512', 'SPEC513', 'SPEC576', 'SPEC511', 'SPEC579', 'SPEC575', 'SPEC539', 'SPEC523', 'SPEC526', 'SPEC532', 'SPEC533', 'SPEC596', 'SPEC595', 'SPEC546', 'SPEC559', 'SPEC560', 'SPEC567', 'SPEC545', 'SPEC544', 'SPEC568', 'SPEC540', 'SPEC542'] 
sel_6 = ['SPEC602', 'SPEC603', 'SPEC601', 'SPEC606', 'SPEC605', 'SPEC664', 'SPEC667', 'SPEC662', 'SPEC647', 'SPEC645', 'SPEC642', 'SPEC643', 'SPEC625', 'SPEC698', 'SPEC695', 'SPEC696', 'SPEC690', 'SPEC693', 'SPEC617', 'SPEC610', 'SPEC618', 'SPEC650', 'SPEC653', 'SPEC638', 'SPEC655', 'SPEC671', 'SPEC630', 'SPEC636', 'SPEC635', 'SPEC678', 'SPEC674', 'SPEC673', 'SPEC681', 'SPEC686'] 
sel_7 = ['SPEC732', 'SPEC733', 'SPEC731', 'SPEC737', 'SPEC734', 'SPEC722', 'SPEC738', 'SPEC735', 'SPEC729', 'SPEC728', 'SPEC725', 'SPEC718', 'SPEC726', 'SPEC716', 'SPEC701', 'SPEC700', 'SPEC706', 'SPEC705', 'SPEC717']

#fors2_raw_lst_test= sel_1+sel_2+sel_3+sel_4+sel_5+sel_6+sel_7
fors2_raw_lst_test= l0

final_cleaned_list =  ['SPEC509', 'SPEC726', 'SPEC223', 'SPEC605', 'SPEC501', 'SPEC526', 'SPEC524', 'SPEC376', 'SPEC728', 'SPEC442', 'SPEC488', 'SPEC218', 'SPEC583', 'SPEC191', 'SPEC178', 'SPEC480', 'SPEC700', 'SPEC643', 'SPEC610', 'SPEC297', 'SPEC517', 'SPEC160', 'SPEC722', 'SPEC527', 'SPEC516', 'SPEC571', 'SPEC560', 'SPEC512', 'SPEC575', 'SPEC576', 'SPEC414', 'SPEC695', 'SPEC419', 'SPEC418', 'SPEC690', 'SPEC717', 'SPEC513', 'SPEC733', 'SPEC109', 'SPEC737', 'SPEC267', 'SPEC242', 'SPEC523', 'SPEC455', 'SPEC511', 'SPEC453', 'SPEC474', 'SPEC494', 'SPEC504', 'SPEC184', 'SPEC653', 'SPEC638', 'SPEC655', 'SPEC363', 'SPEC493', 'SPEC630', 'SPEC222', 'SPEC678', 'SPEC385', 'SPEC382', 'SPEC539', 'SPEC282', 'SPEC280', 'SPEC667', 'SPEC567', 'SPEC693', 'SPEC460', 'SPEC461', 'SPEC540', 'SPEC468', 'SPEC542']

#fors2_raw_lst_full=final_cleaned_list
fors2_raw_lst_full=fors2_raw_lst_test

red    =  ['SPEC509', 'SPEC605', 'SPEC501', 'SPEC526', 'SPEC524', 'SPEC376', 'SPEC504', 'SPEC488', 'SPEC218', 'SPEC583', 'SPEC667', 'SPEC700', 'SPEC517', 'SPEC576', 'SPEC511', 'SPEC695', 'SPEC223', 'SPEC453', 'SPEC493', 'SPEC222', 'SPEC494', 'SPEC282', 'SPEC560']
medium =  ['SPEC726', 'SPEC527', 'SPEC523', 'SPEC728', 'SPEC442', 'SPEC178', 'SPEC539', 'SPEC516', 'SPEC571', 'SPEC512', 'SPEC575', 'SPEC418', 'SPEC690', 'SPEC513', 'SPEC737', 'SPEC242', 'SPEC455', 'SPEC474', 'SPEC638', 'SPEC655', 'SPEC382', 'SPEC567', 'SPEC460', 'SPEC542']
blue   =   ['SPEC722', 'SPEC191', 'SPEC480', 'SPEC643', 'SPEC297', 'SPEC160', 'SPEC414', 'SPEC419', 'SPEC717', 'SPEC693', 'SPEC733', 'SPEC109', 'SPEC610', 'SPEC267', 'SPEC363', 'SPEC184', 'SPEC653', 'SPEC630', 'SPEC678', 'SPEC385', 'SPEC280', 'SPEC461', 'SPEC540', 'SPEC468']


#####################################################
#####################################################
#####################################################
path_raw_jct='/home/enuss/00_labo/lsst/photoz/fors2/seds/'
cat = fits.open('/home/enuss/00_labo/lsst/photoz/fors2/data/fors2_catalogue.fits')[1]
sl_path='/home/enuss/00_labo/lsst/photoz/sl04/'
path_ana='/home/enuss/00_labo/lsst/photoz/edmond_fors2/'
path_out_jct='/home/enuss/00_labo/lsst/photoz/edmond_fors2/dataset_edmond/raw_SEDs/jct_redshifted/' #jct raw dat in SL format

ext        =os.environ['EXT_LAW']
ana_type   =os.environ['ANA_TYPE']
run_type   =os.environ['RUN_TYPE']
base_tag   =os.environ['BASE_TAG']
plot_type  =os.environ['PLOT_TYPE']
config_tag=os.environ['CONFIG_TAG']

if ana_type=='fors2_test' :
  self.lst=fors2_raw_lst_test

if (ana_type=='fors2_raw') :
  if(run_type=='full'):
    lst=fors2_raw_lst_full
  if(run_type=='test'):
    lst=fors2_raw_lst_test
    #raw_assos=fors2_raw_assos_test

if ana_type=='fors2' :
  if(run_type=='full'):
    lst=fors2_lst_full
  if(run_type=='test'):
    lst=fors2_lst_test

if (ana_type=='brown_rebuild' or ana_type=='brown_wide')  :
  if(run_type=='full'):
    lst=brown_lst_full
  if(run_type=='illustrative'):
    lst=brown_lst_illustrative
  if(run_type=='test'):
    lst=brown_lst_test

#---------------------------------------------
#initial spectra used as inputs for starlight:
if ana_type=='fors2_test' :
  path_data='/home/enuss/00_labo/lsst/photoz/edmond_fors2/dataset_edmond/raw_SEDs/test_jct/'
  path_truncated=path_data #for raw data we do not truncate


if ana_type=='fors2_raw' :
  path_data='/home/enuss/00_labo/lsst/photoz/edmond_fors2/dataset_edmond/raw_SEDs/jct_redshifted/' #for raw data we do not truncate
  path_truncated=path_data #for raw data we do not truncate

if ana_type=='fors2' :
  path_truncated='/home/enuss/00_labo/lsst/photoz/edmond_fors2/SPEC/'
  path_data=path_truncated #we do not truncate

if ana_type=='brown_rebuild' :
  path_truncated=path_brown+'truncated_brown_spectra/'
  path_data=path_brown_data #we do not truncate

if ana_type=='brown_wide' :
  path_truncated=path_brown+'truncated_brown_wide_spectra/'
  path_data=path_brown_data#we do not truncate

#---------------------------------------------
# Define basedir for SL:
if (base_tag=='BC03N'):
   path_base_spectra =sl_path+'BasesDir/' #for Base.BC03.N
elif (base_tag=='BC03S'):
   path_base_spectra =sl_path+'BasesDir/' #for Base.BC03.S
elif (base_tag=='JM'):
   path_base_spectra =sl_path+'Jorge/BasesDir/'         #for JM_Base.BC03.Vbm
#---------------------------------------------
# Set SL path:


input_files_path=sl_path+'ext/'+ana_type+'/'
if (os.path.exists(input_files_path) != True):
  os.mkdir(input_files_path)
input_files_path=sl_path+'ext/'+ana_type+'/'+base_tag+'/'
if (os.path.exists(input_files_path) != True):
  os.mkdir(input_files_path)
input_files_path=sl_path+'ext/'+ana_type+'/'+base_tag+'/'+config_tag+'/'
if (os.path.exists(input_files_path) != True):
  os.mkdir(input_files_path)
input_files_path=sl_path+'ext/'+ana_type+'/'+base_tag+'/'+config_tag+'/'+'/'+ext+'/'
if (os.path.exists(input_files_path) != True):
  os.mkdir(input_files_path)
input_files_path=sl_path+'ext/'+ana_type+'/'+'/'+base_tag+'/'+config_tag+'/'+ext+'/input_config_file/'
if (os.path.exists(input_files_path) != True):
  os.mkdir(input_files_path)

ana_path=sl_path+'ext/'+ana_type+'/'+base_tag+'/'+config_tag+'/'+ext+'/'
ext_path=ana_path+'/extended_spectra/'
path_brown="/home/enuss/00_labo/lsst/photoZ/brown_atlas/123/"
path_brown_data=path_brown+'605/'
ext='_'+ext
  
path_SL           =ana_path+'output_sl/'

path_rebuild_BC   =ana_path       +'output_rebuild_BC/'
path_BC           =path_rebuild_BC+'full_spectra/'          #BC spectra without extinction
path_BC_ext       =path_rebuild_BC+'full_spectra_ext/'      #BC spectra with extinction
path_BC_pop_sp    =path_rebuild_BC+'population_spectra/'    #BC individual spectra without extinction
path_BC_pop_sp_ext=path_rebuild_BC+'population_spectra_ext/'#BC individual spectra with extinction
path_extinction_models='/home/enuss/00_labo/lsst/photoz/lephare/lephare_dev/ext/'

if (os.path.exists(path_SL) != True):
  os.mkdir(path_SL)
if (os.path.exists(path_rebuild_BC) != True):
  os.mkdir(path_rebuild_BC)
if (os.path.exists(path_BC) != True):
  os.mkdir(path_BC)
if (os.path.exists(path_BC_ext) != True):
  os.mkdir(path_BC_ext)
if (os.path.exists(path_BC_pop_sp) != True):
  os.mkdir(path_BC_pop_sp)
if (os.path.exists(path_BC_pop_sp_ext) != True):
  os.mkdir(path_BC_pop_sp_ext)

class sl_out(object):
  def __init__(self,id,label=""):
      file_in =path_SL+'SPEC'+id+ext+'.txt'
      ll=open(file_in).readlines()
      # utiles pour les calculs d'extinction :
      l=ll[14].lstrip()
      q_norm=float(l.split(' ')[0])
      l=ll[22].lstrip()
      l_norm=float(l.split(' ')[0])
      l=ll[59].lstrip()
      AV_min=float(l.split(' ')[0])
      l=ll[25].lstrip()
      fobs_norm=float(l.split(' ')[0])
      A_l_norm=AV_min*q_norm
      l=ll[49].lstrip()
      chi2=float(l.split(' ')[0])
      l=ll[50].lstrip()
      adev=float(l.split(' ')[0])

      start=63
      if (base_tag=='BC03N'):
        n_star_bc=45 #number of BC stars for Base.BC03.N
      elif (base_tag=='BC03S'):
          n_star_bc=150 #number of BC stars for Base.BC03.S
      elif (base_tag=='JM'):
          n_star_bc=114 #for JM_Base.BC03.Vbmm :
          # construction de la liste des infos utiles pour reconstruire les spectres BC :
      vec_id  =[]
      vec_frac=[]
      vec_log_ages =[]
      vec_log_lum_mass =[]
      vec_metalicity =[]
      for j in range(start,start+n_star_bc) : #loop over basedir lines in starlight output file
          l=ll[j].lstrip()
          l2=l.split(' ')
          l3=[x for x in l2 if x != '']
          l4=[x.replace('\n','') for x in l3]
          frac=float(l4[1])
          age =float(l4[4])
          metalicity =float(l4[5])
          lum_mass   =float(l4[6])
          if (frac>(1.e-12)):
              vec_id  =np.append(vec_id,j+1-start)
              vec_frac=np.append(vec_frac,frac)
              vec_log_ages =np.append(vec_log_ages ,np.log10(age))
              vec_log_lum_mass =np.append(vec_log_lum_mass ,np.log10(lum_mass))
              vec_metalicity =np.append(vec_metalicity ,metalicity)
      moy_log_ages=sum(np.array(vec_frac)*np.array(vec_log_ages))/sum(vec_frac)
      moy_metalicity=sum(np.array(vec_frac)*np.array(vec_metalicity))/sum(vec_frac)

      self.q_norm=q_norm
      self.l_norm=l_norm
      self.AV_min=AV_min
      self.fobs_norm=fobs_norm
      self.A_l_norm=A_l_norm
      self.chi2=chi2
      self.adev=adev
      self.moy_log_ages=moy_log_ages
      self.moy_metalicity=moy_metalicity
      self.vec_id=vec_id
      self.vec_frac=vec_frac
      self.vec_log_ages=vec_log_ages
      self.vec_log_lum_mass=vec_log_lum_mass
      self.vec_metalicity=vec_metalicity
      
class SED_jct(object):
  def __init__(self,id,label=""):
      z,lines,ra,dec=get_catalog_info(id,cat)
      filename=path_raw_jct+'SPEC'+str(id)+'n.txt'
      self.d=np.loadtxt(filename, unpack=True)
      if (float(z)!=-1) :
        self.wave_tmp=self.d[0]/(1.+z)
      else:
        self.wave_tmp=self.d[0]*0.
      self.flux_tmp=self.d[1]
      self.mask=self.d[2]
      id_mask=np.where(self.mask==0)
      self.wave=self.wave_tmp[id_mask]
      self.flux=self.flux_tmp[id_mask]      
      self.label=label
      self.z=z
      self.lines=lines
      self.ra=ra
      self.dec=dec
  def get_scale(self,bounds=(4150,4250)):
      start=np.searchsorted(self.wave,bounds[0])
      stop=np.searchsorted(self.wave,bounds[1])
      return self.flux[start:stop].mean()

class SED(object): #input SED to SL
  def __init__(self,filename,label=""):
      self.d=np.loadtxt(filename)
      self.wave=self.d[:,0]
      self.flux=self.d[:,1]
      self.label=label
  def smooth(self,size=3):
      return ndimage.filters.gaussian_filter1d(self.flux,size)
  def rescale(self,value):
      self.flux*=value
  def get_scale(self,bounds=(4150,4250)):
      start=np.searchsorted(self.wave,bounds[0])
      stop =np.searchsorted(self.wave,bounds[1])
      return self.flux[start:stop].mean()

class SED_eg(object):
  def __init__(self,filename,label=""):
      self.d=np.loadtxt(filename)
      self.wave=self.d[:,0]
      self.flux=self.d[:,2]
      self.label=label
  def smooth(self,size=3):
      return ndimage.filters.gaussian_filter1d(self.flux,size)
  def rescale(self,value):
      self.flux*=value
  def get_scale(self,bounds=(4150,4250)):
      start=np.searchsorted(self.wave,bounds[0])
      stop=np.searchsorted(self.wave,bounds[1])      
      return self.flux[start:stop].mean()
      
def get_catalog_info(spec, cat):
    try:
        spec = int(spec)
    except:
        z=-1
        lines='redshift unknown'
    if spec in cat.data['ID']:
        catid=(cat.data['ID']==spec)
        z=cat.data[catid]['z'][0]
        lines=cat.data[catid]['Lines'][0]
        ra =cat.data[catid]['RAJ2000'][0]
        dec=cat.data[catid]['DEJ2000'][0]
    else:
         z=-1
         lines='not in catalog'
         ra=0.
         dec=0.
    return z,lines,ra,dec

    
def file2list(filename,cmt="#"):
    ll=open(filename).readlines()
    res=[]
    for i,l in enumerate(ll):
        if l[0]!='#':
            res.append(l.strip("\n"))
    return res

#####################################################
#####################################################
#####################################################
