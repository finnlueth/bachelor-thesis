#!/usr/bin/mawk -f

# mawk -f check_scop_files.awk <lookup_file> <search_path>
# Example: mawk -f check_scop_files.awk ../data/scop_lookup.fix.tsv ./scop-pdb

# Check if we have all required arguments
BEGIN {
    if (ARGC != 3) {
        print "Usage: mawk -f check_scop_files.awk <lookup_file> <search_path>"
        exit 1
    }
    search_path = ARGV[2]
    ARGV[2] = ""  # Clear the second argument so awk doesn't try to process it as a file
    FS = "\t"  # Set field separator to tab for TSV file
    missing_count = 0
}

# Process lookup file and store filenames
{
    if (NF >= 1) {  # Ensure we have at least 1 field
        files_to_check[$1] = 1  # Store the filename from first column
    }
}

END {
    # Check each filename from lookup against the search path
    for (filename in files_to_check) {
        cmd = "test -f " search_path "/" filename
        if (system(cmd) != 0) {
            print "Missing file: " search_path "/" filename
            missing_count++
        }
    }
    
    print "\nSummary:"
    print "Total entries in lookup file: " length(files_to_check)
    print "Missing files: " missing_count
    print "Found files: " (length(files_to_check) - missing_count)
}