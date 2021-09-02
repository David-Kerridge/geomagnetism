# -*- coding: utf-8 -*-
"""
A series of functions designed to help in basic magnetic field data reading
and manipulation.

1. IAGA2002_Header_Reader
2. IAGA2002_Data_Reader

The, assuming IAGA2002 format files:

3. load_year: read one-minute data for a given observatory and year
4. readBmin: one-minute data for a given observatory and date range
   (effectively supersedes 'load_year')
5. read_obs_hmv: hourly mean values for a given observatory and range of years
6. read_obs_hmv_declination: hourly mean values of declination
7. read_obs_ann_mean: annual means for one observatory from a 'BGS format' file
8. obs_ann_means_one_year: annual means for one year from all observatories
9. dh2xy: calculate X and Y from D and H
10. _mag_el_text: defines a dictionary of terms for geomagnetic data
11. _mag_el_fns: a dictionary of functions to calculate geomagnetic field
    elements from other known elements
12. els_calc: to compute all geomagnetic field elements given X, Y, Z
13. els_calc_sv: to calculate rate of change of geomagnetic field elements
    given X, Y, Z and Xdot, Ydot, Zdot
14. els_prn: print results of geomagnetic field element calculations neatly
15. els_prn_sv: neat printing of secular variation calculations
16. dictify: convert XML to python dictionary
17. list_INTERMAGNET_observatories: web request for list of IM observatories
18. get_INTERMAGNET_data: retieve data for a given observatory and date range
19. get_INTERMAGNET_datafile: retrieve a single data file for an observatory

OLder functions now superseded, but retained in the library 'for the record'

1. xyz2dhif: calculate D, H, I and F from X, Y and F
2. xyz2dhif_sv: similar to (1) for rate of change of the field

Notes for improvements:
-----------------------
20 Feb 2021
-----------
Might need to add parameter dtype to IAGA Reader to handle different
INTERMAGNET file types (d=definitive, p=provisional, q=quasi-definitive)

Authors
------
David Kerridge
INTERMAGNET web services functions: adapted from code by Simon Flower

"""

import os
import pandas as pd
import numpy as np
from collections import namedtuple
import requests
import json
import xml.etree.ElementTree as ET
from copy import copy
from datetime import datetime

r2d = np.rad2deg
d2r = np.deg2rad

#==============================================================================

def IAGA2002_Header_Reader(IAGA2002_file):
    
    """
    This function counts the header and comment rows in an IAGA 2002 format
    file. It is designed to cope with the number of header lines being either
    12 or 13, and an arbitrary number of comment lines (including none).
    
    (The IAGA2002 format was last revised in June 2015 to allow an optional
    thirteenth header line 'Publication date'.   
    Ref: https://www.ngdc.noaa.gov/IAGA/vdat/IAGA2002/iaga2002format.html)
    
    The rows of data are preceded by a row of column headers starting with
    "DATE" in columns 0:3. This string cannot occur earlier in the file, so
    detecting the first occurence of this string may be used to count the total 
    number of header and comment lines.
          
    This function may be useful to define the number of rows to skip 
    (n_header + n_comment) in another function designed to read in the data. 
    While it is rather cumbersome, when reading in a long sequence of IAGA2002
    files, the 'safety first' approach would be to call this function for each
    file in case the number of header lines changes within the sequence of 
    files.
      
    Parameters
    ----------
    IAGA2002_file: string
        The full path and file name for the IAGA2002 data file
    
    Returns
    ------
    A tuple: 
        n_header: the integer number of header rows   
        n_comment: the integer number ofcomment rows
        headers, a dictionary containing the information in the headers
    
    Dependencies
    ------------
    pandas
      
    """
    
    COMMENT_STR   = '#'
    DATE_STR      = 'DATE'
    head          = 4*' '
    n_header = 0
    n_lines  = 0
    headers  = {}
    
    with open(IAGA2002_file) as ofile:
        while head[0:4] != DATE_STR:
            head = next(ofile)
            if head[1] != COMMENT_STR:
                key  = head[0:24].strip()
                val  = head[24:69].strip()
                headers[key] = val
                n_header += 1
            n_lines += 1

    headers.pop(key)  # Remove the data column header line from the dictionary
    n_comment  = n_lines-n_header     # The number of comment lines
    n_header  -= 1                    # The number of header lines
    return (n_header, n_comment, headers)

#==============================================================================

def IAGA2002_Data_Reader(IAGA2002_file):
    
    """
    This function reads the data in an IAGA 2002 format file into a pandas
    dataframe.
      
    Parameters
    ----------
    IAGA2002_file: string
        The full path and file name for the IAGA2002 data file
    
    Returns
    ------
    A pandas dataframe: 
        df: the data with a datetime index and column labels from the
            IAGA2002 file
    
    Dependencies
    ------------
    pandas
      
    """
    
# Read the header and comment lines at the top of the file to find how many 
# rows to skip before reading the data
    header = IAGA2002_Header_Reader(IAGA2002_file)
    nskip  = header[0]+header[1]

# Read the data into a pandas dataframe (an IAGA2002 file has 'DATE' and 'TIME'
# as the first two column labels.) There's a trailing '|' on the column header
# line which is interpreted as the header for a column of nans and this 
# property is used to delete it. 

    DT_INDEX = 'DATE_TIME'
    df = pd.read_csv(IAGA2002_file, 
                     delim_whitespace=True,
                     skiprows=nskip,
                     parse_dates=[DT_INDEX.split('_')],
                     index_col=DT_INDEX)
    df.dropna(inplace=True, axis=1)
    
    return(df)

#==============================================================================

def load_year(obscode, year, dname):
    
    """
    Read in the daily 1-min files for a calendar year.
    
    Parameters
    ----------
        obscode: string
            Observatory code e.g. ESK
        year: int/string
            Desired year to load
        dname: string
            Directory containing the files for that year
    
    Returns
    -------
        DataFrame
        
    Dependencies
    ------------
    pandas, os
    
    """
    
    dates_in_year = pd.date_range(start=f'{year}-01-01', \
                                  end=f'{year}-12-31', freq='D')
    df = pd.DataFrame()
    for date in dates_in_year:
        ymd = date.strftime('%Y%m%d')
        file_name = f'{obscode}{ymd}dmin.min'
        file_path = os.path.join(dname, file_name)
        df = df.append(IAGA2002_Data_Reader(file_path))
    return df

#==============================================================================

def readBmin(obscode, daterange, dname):
    
    """
    Read in daily 1-min files (IAGA2002 format) over a range of dates
    
    Parameters
    ----------
        obscode: str
            Observatory code e.g. ESK
        daterange: list(two items), str
            start date and end date
        dname: str
            Name of directory containing the files for the observatory
            (The directory structure is assumed to be of the form
            'dname/year/one-min-files'. This is used to read files over a 
            year boundary.)
    
    Returns pandas DataFrame
    -------
    df
        
    Dependencies
    ------------
    pandas, os
    
    """
    df = pd.DataFrame()
    dates = pd.date_range(start=daterange[0], end=daterange[1], freq='D')
    
    for d in dates:
        syr   = str(d.year)
        ymd = d.strftime('%Y%m%d')
        file_name = f'{obscode}{ymd}dmin.min'
        file_path = os.path.join(dname, syr, file_name)
        df = df.append(IAGA2002_Data_Reader(file_path))
    return df

#==============================================================================

def read_obs_hmv(obscode, year_st, year_fn, folder):
    """Read in observatory annual mean files in IAGA2002 format.
    
    This function reads the hourly mean value data in yearly IAGA2002 format
    files into a pandas dataframe. (Note: The data may be reported in different
    ways in different years (e.g. DFHZ, FXYZ).)
      
    Input parameters
    ---------------
    obscode: the IAGA observatory code: string (3 or 4 characters)
    year_st: the start year for the data request
    year_fn: the final year for the data request
    folder : the location of the yearly hmv files
    
    Output
    ------
    A pandas dataframe: datareq
    This has columns of X, Y and Z data (only) and keeps the datetime index
    from the IAGA2002 files
    
    Dependencies
    ------------
    pandas
    
    """
    OBSY   = obscode.upper()
    obsy   = obscode.lower()
    # Read in the observatory data one year file at a time and construct 
    # filenames
    datareq = pd.DataFrame()        
    for year in range(year_st, year_fn+1):
        ystr    = str(year)
        file    = obsy + ystr + 'dhor.hor'
        fpf     =  folder + file
        tmp     = IAGA2002_Data_Reader(fpf)
        tmp.columns = [col.strip(OBSY) for col in tmp.columns]
        if('D' in tmp.columns):
            xvals, yvals  = dh2xy(tmp['D'], tmp['H'])
            tmp['X'] = xvals.round(decimals=1)
            tmp['Y'] = yvals.round(decimals=1)
        datareq = datareq.append(tmp[['X','Y', 'Z']])
    return(datareq)

#==============================================================================    

def read_obs_hmv_declination(obscode, year_st, year_fn, folder):
    
    """
    Read (or calculate) declination from hourly mean files in IAGA2002 format.

    This function reads the hourly mean value data in yearly IAGA2002 format
    files into a pandas dataframe for the specified observatory between 
    year_st and year_fn. Note that D is reported in angular units of minutes 
    of arc (and not degrees) in this file format.

    Input parameters
    ---------------
    obscode: the IAGA observatory code: string (3 or 4 characters)
    year_st: the start year for the data request
    year_fn: the final year for the data request
    folder : the location of the yearly hmv files

    Output
    ------
    A pandas dataframe: datareq
    This has columns for datetime and declination
    
    Dependencies
    ------------
    pandas

    Local Dependencies
    ----------------
    none

    Revision date
    -------------
    24/06/19 (Grace Cox)

    """

    OBSY   = obscode.upper()
    obsy   = obscode.lower()
    # Read in the observatory data one year file at a time and construct 
    # filenames
    datareq = pd.DataFrame()
    for year in range(year_st, year_fn+1):
        ystr    = str(year)
        file    = obsy + ystr + 'dhor.hor'
        fpf     =  folder + '/' + file
        tmp     = IAGA2002_Data_Reader(fpf)
        tmp.columns = [col.strip(OBSY) for col in tmp.columns]
        tmp = tmp.replace(99999.00, np.nan)
        # Calculate D (in degrees) if not given in the file
        if('D' not in tmp.columns):
            dvals, hvals, ivalsm, fvals=xyz2dhif(tmp['X'], tmp['Y'], tmp['Z'])
            tmp.insert(loc=1, column='D', value=dvals.values)
        else:
            # Convert the reported values to degrees
            tmp['D'] = tmp.D.values/60.0
        datareq = datareq.append(tmp[['D']])
    return(datareq)

#==============================================================================

def read_obs_ann_mean(obscode, filename):
    
    """Read in the annual mean values for a single observatory.
      
    Input parameters
    ---------------
    obscode:  the IAGA observatory code: string (3 or 4 characters)
    filename: the file of observatory annual mean values
    
    Output
    ------
    A pandas dataframe with the observatory annual mean values for all years
    available (indexed by year)
    
    Dependencies
    ------------
    pandas
         
    Revision date
    -------------
    30 Jan 2019

    """
    count = 0
    with open(filename) as ofile:
        for line in ofile:
            if count==0:
                df = pd.DataFrame(columns=line.split())
                count += 1
            if str(obscode) in line:
                df.loc[count] = line.split()
                count +=1
    return(df)

#==============================================================================
    
def obs_ann_means_one_year(year, filename):
    """Read in the annual mean values for all observatories in a given year.
      
    Input parameters
    ---------------
    obscode: the IAGA observatory code: string (3 or 4 characters)
    year:    the years are integers, (almost all) in the format yyyy.5
    
    Output
    ------
    A pandas dataframe with the observatory annual mean values for the year
    requested. 
    
    Dependencies
    ------------
    pandas
         
    Revision date
    -------------
    30 Jan 2019

    """
    
    count = 0
    with open(filename) as ofile:
        for line in ofile:
            if count==0:
                df = pd.DataFrame(columns=line.split())
                count += 1
            if str(year) in line:
                df.loc[count] = line.split()
                count +=1
    return(df)

#==============================================================================
   
def dh2xy(d, h):
    """
    Calculate X and Y from D (in units of 0.1 minutes) and H as reported 
    in IAGA2002format files.
      
    Input parameters
    ---------------
    D: declination (0.1 minutes)
    H: horizintal intensity (nT)
    
    Output
    ------
    A tuple: (X, Y), the North and East geomagnetic field components in nT
    
    Dependencies
    ------------
    numpy
         
    Revision date
    -------------
    30 Jan 2019

    """
    dec = d*np.pi/(180.*60.)
    return (h*np.cos(dec), h*np.sin(dec))

#==============================================================================

def _mag_el_text():
    
    """
    To give easy access to the conventional labels for the 7 geomagnetic field 
    elements for use in printed or graphical outputs. The user is assumed to 
    be aware of the meaning of the abbreviations D, H, Z, X, Y, Z, I, and F.
    However, this function is intended primarily to be called by other
    functions.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    d: a dictionary with the 7 element abbreviations as keys and each value a 
    dictionary with 4 keys:
        name: the full name of the element
        abbr: the element abbreviation letter in parentheses
        unit: the unit of measurement
        svun: the units for secular variation (rate of change)
        
    Examples
    --------
    >>> _mag_el_text()['I']['name']
        'Inclination'
    >>> _mag_el_text()['D']['unit']
        'deg'
        
    Revision date
    -------------
    22 Feb 2021
    
    """
    
    
    d = {
        'D': {'name': 'Declination', 'abbr': 'D', 'unit': 'deg', 
              'svun':'min/y'},
        'H': {'name': 'Horizontal Intensity', 'abbr': 'H', 'unit': 'nT',
              'svun': 'nT/y'},
        'Z': {'name': 'Vertical Intensity', 'abbr': 'Z', 'unit': 'nT',
              'svun': 'nT/y'},
        'X': {'name': 'North Component', 'abbr': 'X', 'unit': 'nT',
              'svun': 'nT/y'},
        'Y': {'name': 'East Component', 'abbr': 'Y', 'unit': 'nT',
              'svun': 'nT/y'},
        'I': {'name': 'Inclination', 'abbr': 'I', 'unit': 'deg', 
              'svun':'min/y'},
        'F': {'name': 'Total Intensity', 'abbr': 'F', 'unit': 'nT',
              'svun': 'nT/y'}
        }
            
    return d

#==============================================================================

def _mag_el_fns(type_='mf'):

    """
    A collection of lambda functions enabling calculation of geomagnetic
    field elements from other elements. There are many permutations, and
    only a few commonly-used conversions are included. The user is assumed to 
    be aware of the meaning of the abbreviations D, H, Z, X, Y, Z, I, and F.
    However, this function is intended primarily to be called by other
    functions.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    d: a dictionary with lambda functions as values
        The keys indicate the functionality. For example key 'F(XYZ)' means
        the value is a function to compute F from X, Y and Z, which should be 
        supplied as the input parameters to the lambda function.
        Input D and I values to the lambda functions are assumed to be in
        degrees and are output in degrees. Other elements are in units of nT.
        
    Dependencies
    ------------
    numpy
    
    Revision date
    -------------
    22 Feb 2021
    
    """
    
    mf_bank = {
        'X(HD)'  : lambda h,d: h*np.cos(d2r(d)),
        'Y(HD)'  : lambda h,d: h*np.sin(d2r(d)),
        'I(ZH)'  : lambda z,h: r2d(np.arctan2(z,h)),
        'F(HZ)'  : lambda h,z: np.sqrt(h*h+z*z),
        'D(YX)'  : lambda y,x: r2d(np.arctan2(y,x)),
        'H(XY)'  : lambda x,y: np.sqrt(x*x+y*y),
        'I(XYZ)' : lambda x,y,z: r2d(np.arctan2(z, np.sqrt(x*x+y*y))),
        'F(XYZ)' : lambda x,y,z: np.sqrt(x*x+y*y+z*z),
        'X(FID)' : lambda f,i,d: f*np.cos(d2r(i))*np.cos(d2r(d)),
        'Y(FID)' : lambda f,i,d: f*np.cos(d2r(i))*np.sin(d2r(d)),
        'H(FI)'  : lambda f,i: f*np.sin(d2r(i)),
        'Z(FI)'  : lambda i,f: f*np.cos(d2r(i))
        }
    
    sv_bank = {
        'Dd(XYH)' : lambda x,xd,y,yd,h: r2d(60*(x*yd-y*xd)/(h*h)),
        'Hd(XYH)' : lambda x,xd,y,yd,h: (x*xd+y*yd)/h,
        'Id(HZF)' : lambda h,hd,z,zd,f: r2d(60*(h*zd-z*hd)/(f*f)),
        'Fd(XYZF)': lambda x,xd,y,yd,z,zd,f: (x*xd+y*yd+z*zd)/f       
              }
    
    if type_=='mf':
        return mf_bank
    elif type_=='sv':    
        return (mf_bank,sv_bank)

#==============================================================================

def els_calc(xyz):
    
    """
    A function to compute the geomagnetic elements D, H, I and F given X, Y  
    and Z. (This function could be developed further to accept other input 
        triplets.)
    
    Parameters
    ----------
    xyz - tuple, floats
        X, Y and Z components of a geomagnetic field vector (in nT)
    
    Returns
    -------
    b: named tuple
        b contains all 7 elements of the geomagnetic field vector which can be
        accessed as, for example b.D for declination
        
    Dependencies
    ------------
    collections
    _mag_el_fns in this module
    
    Revision date
    -------------
    22 February 2021
    
    """
    
    Mag_els = namedtuple('Mag_els', 'X Y Z D H I F')
    Mag_els.__new__.__defaults__ = (None,)*len(Mag_els._fields)
    Mag_els.__doc__ = 'Seven elements of the geomagetic field vector'

    b = Mag_els(*xyz)
    fns = _mag_el_fns()
    b = b._replace(D = fns['D(YX)'](b.Y,b.X),
                   H = fns['H(XY)'](b.X,b.Y),
                   I = fns['I(XYZ)'](b.X, b.Y, b.Z),
                   F = fns['F(XYZ)'](b.X, b.Y, b.Z)
                  )
    return b

#==============================================================================

def els_calc_sv(xyz, xyzdot):
    
    """
    A function to compute the secular variation of geomagnetic elements D, H, 
    I and F given X, Y, Z and their secuar variation values Xdot, Ydot and
    Z dot. 
    
    Parameters
    ----------
    xyz - tuple, floats
        X, Y and Z components of a geomagnetic field vector (in nT)
    xyzdot - tuple, floats
        Xdot, Ydot and Zdot (nT/y)
    
    Returns
    -------
    sv: named tuple
        sv contains all 7 elements of the secular variation of the geomagnetic 
        field vector which can be accessed as, for example sv.Hd for the rate
        of change of horizontal intensity
        
    Dependencies
    ------------
    collections
    _mag_el_fns in this module
    
    Revision date
    -------------
    22 February 2021
    
    """

    Mag_els = namedtuple('Mag_els', 'X Y Z H F')
    Mag_els.__new__.__defaults__ = (None,)*len(Mag_els._fields)
    Mag_els.__doc__ = 'Elements required for computing secular variation'
    
    SV_els = namedtuple('SV_els', 'Xd Yd Zd Dd Hd Id Fd')
    SV_els.__new__.__defaults__ = (None,)*len(SV_els._fields)
    SV_els.__doc__ = 'Seven elements of the secular variation'

    b = Mag_els(*xyz)
    fns, svfns = _mag_el_fns('sv')
    b = b._replace(H = fns['H(XY)'](b.X,b.Y),
                   F = fns['F(XYZ)'](b.X, b.Y, b.Z)
                  )    
    
    sv = SV_els(*xyzdot)
    sv = sv._replace(Dd=svfns['Dd(XYH)'](b.X, sv.Xd, b.Y, sv.Yd, b.H),
                     Hd=svfns['Hd(XYH)'](b.X, sv.Xd, b.Y, sv.Yd, b.H),
                     Fd=svfns['Fd(XYZF)'](b.X, sv.Xd, b.Y, sv.Yd,
                                          b.Z, sv.Zd, b.F))
    sv = sv._replace(Id=svfns['Id(HZF)'](b.H, sv.Hd, b.Z, sv.Zd, b.F))
    
    return sv

#==============================================================================

def els_prn(b, elstr):
    
    """
    A function to print the geomagnetic elements. It is designed to be used in
    conjunction with function els_calc in this module
    
    Parameters
    ----------
    b - named tuple, floats
            - containing 7 elements of a geomagnetic field vector
    elstr - str
        The elements to be printed, in the order in the string
    
    Returns
    -------
    None
        
    Dependencies
    ------------
    _mag_el_text in this module
    
    Revision date
    -------------
    19 Feb 2021
    
    """

    e1txt = _mag_el_text()
    for el in elstr:
        str1 = e1txt[el]['name']
        str2 = e1txt[el]['unit']
        x = b.__getattribute__(el)
        if el in 'DI':
            val  = f'{x:.2f}'
        else:
            val  = f'{x:.1f}'
        print(f'{str1:21}: {val:>8} {str2}')
        
#==============================================================================

def els_prn_sv(sv, elstr):

    """
    A function to print the secular variation of the geomagnetic elements. 
    It is designed to be used in conjunction with function els_calc_sv in 
    this module
    
    Parameters
    ----------
    sv - named tuple, floats
            - containing 7 elements of the rate of change of a geomagnetic
              field vector
    elstr - str
        The elements to be printed, in the order in the string
    
    Returns
    -------
    None
        
    Dependencies
    ------------
    _mag_el_text in this module
    
    Revision date
    -------------
    22 Feb 2021
    
    """

    e1txt = _mag_el_text()
    for el in elstr:
        idx = el+'d'
        str1 = e1txt[el]['name']
        str2 = e1txt[el]['svun']
        x = sv.__getattribute__(idx)
        val  = f'{x:.1f}'
        print(f'{str1:21}: {val:>6} {str2}')
         
#==============================================================================

def dictify(r,root=True):
    """
    Convert XML to a Python dictionary.
    Code taken from: https://stackoverflow.com/questions/2148119    
    """
    if root:
        return {r.tag : dictify(r, False)}
    d=copy(r.attrib)
    if r.text:
        d["_text"]=r.text
    for x in r.findall("./*"):
        if x.tag not in d:
            d[x.tag]=[]
        d[x.tag].append(dictify(x,False))
    return d

#==============================================================================

def list_INTERMAGNET_observatories():
    """
    Return a list of the observatories available from the INTERMAGNET
    web service. The returned value has a complex data structure reflecting
    the structure of the database holding the observatory information.
    
    Returns:
    -------
    A dictionary:
        
    Details:
    --------
    keys = ['GINServices']
    ['GINServices'].keys() = '_text','ObservatoryList','PublicationStateList'])
     
    
    'ObservatoryList' is a list with one element, a dictionary:
        ['GINServices']['ObservatoryList'][0].keys() = ['_text','Observatory']
        
        'Observatory' is a list of dictionaries, one per observatory.
        e.g. ['GINServices']['ObservatoryList'][0]['Observatory'][0].keys() = 
        ['_text', 'Code', 'Name', 'Latitude', 'Longitude'])
    
        ....['Code'] is a list with one element, a dictionary with one entry,
        the observatory IAGA code. (Similarly for ['Name'], ['Latitude'] and
        ['Longitude'])
        
        e.g. ....['Code'][0].values() = dict_values(['ABK'])
        (As, at the time of this comment ABK (Abisko, Sweden) was 
         alphabetically the first observatory in the INTERMAGNET database.)
        
    """

# the base URL for the INTERMAGNET web service
    BASE_URL = "https://imag-data.bgs.ac.uk/GIN/GINServices"
    url = BASE_URL + "?request=GetCapabilities&Format=JSON"
    response = requests.get(url)
    
    if response.status_code != 200: # This means something went wrong.
        raise requests.HTTPError("Error calling INTERMAGNET web site, \
                        http response code was " + str(response.status_code))
	
# convert the JSON response into a python list
    root = ET.fromstring(response.text)
    return dictify(root)

#==============================================================================

def get_INTERMAGNET_data (iaga_code, start_date, n_days, 
                          quality='best-available'):
    
    """
    
    Retrieve one-minute data from the INTERMAGNET web service as a Python 
    dictionary
    
    Parameters:
    ----------
    iaga_code: string
        3-character IAGA code (lower case)
    start_date: string
        date of first day of data ('YYYY-MM-DD')
    n_days: (int)
        number of days of data
    quality: (string)
        one of:
            "reported" - raw data, possibly without baseline
            "adjusted" - raw data with baseline
            "quasi-def" - quasi definitive (see www.intermagnet.org)
            "definitive" - (there is currently no definitive data available
            from the web service, though there are plans to implement this)
            "best-available" - let the service choose
                      
    Returns:
    -------
    A dictionary with Geomagnetic data in Coverage JSON format
    
    """
    
# Convert to web service terminology
    if quality == "best-available":
        quality = "adj-or-rep"
	
# construct a URL for the request
    BASE_URL = "https://imag-data.bgs.ac.uk/GIN/GINServices"
    url = BASE_URL + \
            "?request=GetData&samplesPerDay=minute&observatoryIagaCode=" \
            + iaga_code + \
            "&dataStartDate=" + start_date + \
            "&dataDuration="  + str(n_days) + \
            "&publicationState=" + quality + \
            "&format=CovJSON"
    response = requests.get(url)
    
    if response.status_code != 200: # This means something went wrong
        raise requests.HTTPError("Error calling INTERMAGNET web site, \
                    http response code was " + str(response.status_code))

# convert the response to a dictionary
    return json.loads(response.text)

#==============================================================================

def get_INTERMAGNET_datafile (iaga_code, start_date, n_days, filename,
                              quality='best-available', format_='iaga2002'):
    """
    Retrieve a (single) file of one-minute data in using the INTERMAGNET
    web service
    
    Parameters:
    ----------
    iaga_code: string
        3-character IAGA code (lower case)
    start_date: string
        date of first day of data ('YYYY-MM-DD')
    n_days: (int)
        number of days of data
    quality: (string)
        one of:
            "reported" - raw data, possibly without baseline
            "adjusted" - raw data with baseline
            "quasi-def" - quasi definitive (see www.intermagnet.org)
            "definitive" - (there is currently no definitive data available
            from the web service, though there are plans to implement this)
            "best-available" - let the service choose
    format_:  (string)
        the format of the data - see "Data Service" "format"
        parameter at https://imag-data.bgs.ac.uk/GIN/
    filename: (string)
        destination path and filename for the data
        
    """
    
# convert to web service terminology
    if quality == "best-available":
        quality = "adj-or-rep"
    
    BASE_URL = "https://imag-data.bgs.ac.uk/GIN/GINServices"
    url = BASE_URL + \
            "?request=GetData&samplesPerDay=minute&observatoryIagaCode=" + \
            str(iaga_code) + \
            "&dataStartDate=" + start_date + \
            "&dataDuration=" + str(n_days) + \
            "&publicationState=" + quality + \
            "&format=" + format_
    response = requests.get(url)
    
    if response.status_code != 200: # This means something went wrong       
        raise requests.HTTPError("Error calling INTERMAGNET web site, \
                http response code was " + str(response.status_code))

# Write the data to the file
    with open (filename, "w") as file:
        file.write(response.text)

#==============================================================================


###############################################################################

#====== Some older functions now superseded ===================================


#====== els_calc(xyz) is an alternative to xyz2dhif ===========================

def xyz2dhif(x, y, z):
    """Calculate D, H, I and F from (X, Y, Z)
      
    Input parameters
    ---------------
    X: north component (nT) 
    Y: east component (nT)
    Z: vertical component (nT)
    
    Output
    ------
    A tuple: (D, H, I, F)
    D: declination (degrees)
    H: horizontal intensity (nT)
    I: inclination (degrees)
    F: total intensity (nT)
    
    Dependencies
    ------------
    numpy
          
    Revision date
    -------------
    28 April 2019

    """
    hsq = x*x + y*y
    hoz  = np.sqrt(hsq)
    eff = np.sqrt(hsq + z*z)
    dec = np.arctan2(y,x)
    inc = np.arctan2(z,hoz)
    return((r2d(dec), hoz, r2d(inc), eff))

#========== els_calc_sv(xyz, xyzdot) is an alternative to xyz2dhif_sv =========

def xyz2dhif_sv(x, y, z, xdot, ydot, zdot):
    
    """Calculate secular variation in D, H, I and F from (X, Y, Z) and
    (Xdot, Ydot, Zdot)
      
    Input parameters
    ---------------
    X: float
        North component (nT), and Xdot=dX/dt 
    Y: float
        East component (nT), and Ydot=dY/dt 
    Z: float
        Vertical component (nT), and Zdot=dZ/dt 
    
    Output
    ------
    A tuple: (ddot, hdot, idot, fdot)
        ddot: rate of change of declination (minutes/year)
        hdot: rate of change of horizontal intensity (nT/year)
        idot: rate of change of inclination (minutes/year)
        fdot: rate of change of total intensity (nT/year)
    
    Dependencies
    ------------
    numpy
         
    Revision date
    -------------
    28 April 2019

    """
    h2  = x*x + y*y
    h   = np.sqrt(h2)
    f2  = h2 + z*z
    hdot = (x*xdot + y*ydot)/h
    fdot = (x*xdot + y*ydot + z*zdot)/np.sqrt(f2)
    ddot = r2d((xdot*y - ydot*x)/h2)*60
    idot = r2d((hdot*z - h*zdot)/f2)*60
    
    return (ddot, hdot, idot, fdot)

#==============================================================================