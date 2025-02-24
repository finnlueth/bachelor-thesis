#!/usr/bin/mawk -f

# First pass: Read lookup file and store PDB IDs
FNR==NR {
    pdb_in_lookup[$1] = 1
    next
}

# Second pass: Process directory listing
{
    if ($1 ~ /^d[0-9a-z]/) {  # Only process files starting with 'd'
        pdb_in_dir[$1] = 1
        if (!($1 in pdb_in_lookup)) {
            print "File in directory but not in lookup: " $1
        }
    }
}

END {
    for (pdb in pdb_in_lookup) {
        if (!(pdb in pdb_in_dir)) {
            print "Entry in lookup but no file: " pdb
        }
    }
    
    print "\nSummary:"
    print "Entries in lookup file: " length(pdb_in_lookup)
    print "Files in directory: " length(pdb_in_dir)
    print "Entries missing files: " (length(pdb_in_lookup) - length(pdb_in_dir))
} 