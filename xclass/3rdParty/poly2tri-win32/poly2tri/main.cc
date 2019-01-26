//-------------------------------------------------------------------------/
//Copyright (C) 2003, 2004, 2005, ALL RIGHTS RESERVED.
//Centre for Sys. Eng. & App. Mech.           FEMAGSoft S.A.
//Universite Cathalique de Louvain            4, Avenue Albert Einstein
//Batiment Euler, Avenue Georges Lemaitre, 4  B-1348 Louvain-la-Neuve
//B-1348, Louvain-la-Neuve                    Belgium
//Belgium
//-------------------------------------------------------------------------/
//
//Name:         main.cc (main polygon triangulation funciton by sweep line 
//              algorithm)
//Author:       Liang, Wu (wu@mema.ucl.ac.be, wuliang@femagsoft.com)
//Modified:     10/2005.
//-------------------------------------------------------------------------/

#include "geometry.h"
#include "parse.h"

int main(int argc, char **argv)
{
	Options options;
        parse(argc, argv, options);

        string filename=argv[options.fileindex]; //input bdm file name; 	
        Polygon poly(filename,options.parsebdm);
	poly.setDebugOption(options.debug);      //set debug flag;
       	
	poly.triangulation();                    //main triangulation function
	
	                                         //output results;   
	if(options.showme) poly.saveAsShowme();
	if(options.metapost) poly.saveAsMetaPost();
	if(options.tecplot) poly.saveAsTecplot();

	return 1;
}





