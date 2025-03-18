#!/usr/bin/mawk -f 
BEGIN{OFS="\t";
      print "NAME","CATH_CODE","CLASS","ARCH","TOP","HOM","FP","CLASSCNT","ARCHCNT","TOPCNT","HOMCNT";
}
FNR==NR{
    # Store full CATH code
    cathCode[$1]=$2;
    
    # Split CATH code into components
    split($2, cath, "\\.");
    if(length(cath) == 4) {
        id2class[$1]=cath[1];    # Class (e.g., 3 for mixed α/β)
        id2arch[$1]=cath[1]"."cath[2];    # Architecture (e.g., 3.40)
        id2top[$1]=cath[1]"."cath[2]"."cath[3];    # Topology (e.g., 3.40.50)
        id2hom[$1]=$2;    # Full CATH code (e.g., 3.40.50.300)
        
        # Count occurrences at each level
        classCnt[id2class[$1]]++;
        archCnt[id2arch[$1]]++;
        topCnt[id2top[$1]]++;
        homCnt[id2hom[$1]]++;
    }
    next
} 
!($1 in id2hom) {next}
!($2 in id2hom) {foundFp[$1]++; next}
$1 == $2 {next} # skip self hit

# Check relationships in order of specificity (most specific first)
foundFp[$1] < 1 && id2hom[$1] == id2hom[$2] { foundHom[$1]++; next }  # Same homologous superfamily
foundFp[$1] < 1 && id2top[$1] == id2top[$2] { foundTop[$1]++; next }  # Same topology/fold
foundFp[$1] < 1 && id2arch[$1] == id2arch[$2] { foundArch[$1]++; next }  # Same architecture
foundFp[$1] < 1 && id2class[$1] == id2class[$2] { foundClass[$1]++; next }  # Same class
foundFp[$1] < 1 { foundFp[$1]++; next }  # Different class

END{ 
   for(i in id2hom){
      if(id2hom[i] != "" && homCnt[id2hom[i]] > 1 && 
         topCnt[id2top[i]] - homCnt[id2hom[i]] > 0 && 
         archCnt[id2arch[i]] - topCnt[id2top[i]] > 0 && 
         classCnt[id2class[i]] - archCnt[id2arch[i]] > 0){	   
        homVal=foundHom[i]/(homCnt[id2hom[i]] - 1);
        topVal=foundTop[i]/(topCnt[id2top[i]] - (homCnt[id2hom[i]] - 1));
        archVal=foundArch[i]/(archCnt[id2arch[i]] - (topCnt[id2top[i]] - 1));
        classVal=foundClass[i]/(classCnt[id2class[i]] - (archCnt[id2arch[i]] - 1));
        fpCnt = (foundFp[i] == "") ? 0 : foundFp[i]; 
        print i,cathCode[i],classVal,archVal,topVal,homVal,fpCnt,classCnt[id2class[i]],archCnt[id2arch[i]],topCnt[id2top[i]],homCnt[id2hom[i]];
      }
   }  
}
