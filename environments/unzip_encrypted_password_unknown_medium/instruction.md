A colleague encrypted several important project files in a ZIP archive before leaving the company, but forgot to document the password. The IT department has provided you with some context: the password was related to the project codename and followed the company's password policy from 2019.

You have access to:
- The encrypted archive: `/home/user/secure_backup.zip`
- A partial password hint file: `/home/user/password_hints.txt`
- Historical project documentation: `/home/user/project_docs/`

Your task is to recover the contents of the encrypted ZIP archive. The password is known to be:
- Between 8-12 characters long
- Contains at least one number
- Based on project-related terminology found in the documentation
- May include common password patterns (year suffixes, special character substitutions)

Once you successfully extract the archive, you need to save the recovered password to a file for future reference.

**Output Requirements:**

Save your solution to: `/home/user/solution.txt`

The file must contain exactly one line with the password that successfully unlocked the archive.

**Example output format:**
```
ProjectAlpha2019
```

**Success Criteria:**
- The solution file exists at `/home/user/solution.txt`
- The file contains the correct password (single line, no extra whitespace)
- The password successfully extracts all files from the encrypted archive
- All files from the archive are extracted to the current working directory
