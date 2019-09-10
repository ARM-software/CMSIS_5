/*

Build the table with the platform, compiler and core names.

*/
/*
select NB,CATEGORY.category,NAME,CYCLES,PLATFORM.platform,CORE.core,COMPILERKIND.compiler,COMPILER.version,DATE 
  from BasicBenchmarks
  INNER JOIN CATEGORY USING(categoryid)
  INNER JOIN PLATFORM USING(platformid)
  INNER JOIN CORE USING(coreid)
  INNER JOIN COMPILER USING(compilerid)
  INNER JOIN COMPILERKIND USING(compilerkindid)
  ;

*/

/* 

Compute the max cycles for a test configuration (category + name)

*/
/*
select NAME,max(CYCLES),PLATFORM.platform,CORE.core,COMPILERKIND.compiler,COMPILER.version
  from BasicBenchmarks
  INNER JOIN CATEGORY USING(categoryid)
  INNER JOIN PLATFORM USING(platformid)
  INNER JOIN CORE USING(coreid)
  INNER JOIN COMPILER USING(compilerid)
  INNER JOIN COMPILERKIND USING(compilerkindid)
  GROUP BY NAME,category
  ;
*/

/*

Get last values

*/

/*

Better to use the ON syntax than the USING syntax.
See diff.sql for example

*/

select NB,CATEGORY.category,NAME,CYCLES,PLATFORM.platform,CORE.core,COMPILERKIND.compiler,COMPILER.version,DATE 
  from BasicBenchmarks
  INNER JOIN CATEGORY USING(categoryid)
  INNER JOIN PLATFORM USING(platformid)
  INNER JOIN CORE USING(coreid)
  INNER JOIN COMPILER USING(compilerid)
  INNER JOIN COMPILERKIND USING(compilerkindid)
  WHERE DATE BETWEEN datetime('now','localtime','-10 minutes') AND datetime('now', 'localtime');
