BEGIN {
	count = 0;
}

count += $1;

END {
	printf("Found %d lines\n", count);
}
