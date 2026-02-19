# Slow Query Log file started at 2024-01-15 08:00:00

# Time: 2024-01-15T08:15:23.456789Z
# User@Host: app_user[app_user] @ localhost []
# Query_time: 3.245678  Lock_time: 0.000123 Rows_sent: 45  Rows_examined: 125000
SET timestamp=1705305323;
SELECT * FROM users WHERE email = 'john.doe@example.com';

# Time: 2024-01-15T08:22:15.123456Z
# User@Host: app_user[app_user] @ localhost []
# Query_time: 4.567890  Lock_time: 0.000089 Rows_sent: 120  Rows_examined: 125000
SET timestamp=1705305735;
SELECT id, username, email FROM users WHERE email LIKE '%@gmail.com';

# Time: 2024-01-15T08:35:42.987654Z
# User@Host: app_user[app_user] @ localhost []
# Query_time: 2.123456  Lock_time: 0.000045 Rows_sent: 1  Rows_examined: 125000
SET timestamp=1705306542;
SELECT * FROM users WHERE email = 'jane.smith@example.com' AND status = 'active';

# Time: 2024-01-15T09:12:33.234567Z
# User@Host: app_user[app_user] @ localhost []
# Query_time: 1.890123  Lock_time: 0.000034 Rows_sent: 10  Rows_examined: 98000
SET timestamp=1705308753;
SELECT COUNT(*) FROM users WHERE status = 'inactive';

# Time: 2024-01-15T09:45:18.345678Z
# User@Host: app_user[app_user] @ localhost []
# Query_time: 5.678901  Lock_time: 0.000156 Rows_sent: 234  Rows_examined: 125000
SET timestamp=1705310718;
SELECT * FROM users WHERE created_at > '2024-01-01' ORDER BY created_at DESC;

# Time: 2024-01-15T10:22:45.456789Z
# User@Host: app_user[app_user] @ localhost []
# Query_time: 3.789012  Lock_time: 0.000098 Rows_sent: 1  Rows_examined: 125000
SET timestamp=1705312965;
SELECT username, email FROM users WHERE email = 'alice.johnson@example.com';

# Time: 2024-01-15T11:05:27.567890Z
# User@Host: app_user[app_user] @ localhost []
# Query_time: 2.890123  Lock_time: 0.000067 Rows_sent: 50  Rows_examined: 125000
SET timestamp=1705315527;
SELECT * FROM users WHERE status = 'active' AND created_at > '2024-01-10';

# Time: 2024-01-15T11:38:52.678901Z
# User@Host: app_user[app_user] @ localhost []
# Query_time: 4.123456  Lock_time: 0.000134 Rows_sent: 89  Rows_examined: 125000
SET timestamp=1705317532;
SELECT id, username FROM users WHERE created_at BETWEEN '2023-12-01' AND '2024-01-15';

# Time: 2024-01-15T12:15:33.789012Z
# User@Host: app_user[app_user] @ localhost []
# Query_time: 3.456789  Lock_time: 0.000112 Rows_sent: 1  Rows_examined: 125000
SET timestamp=1705319733;
SELECT * FROM users WHERE email = 'bob.wilson@example.com';

# Time: 2024-01-15T13:42:18.890123Z
# User@Host: app_user[app_user] @ localhost []
# Query_time: 2.567890  Lock_time: 0.000078 Rows_sent: 150  Rows_examined: 125000
SET timestamp=1705324938;
SELECT username, email, status FROM users WHERE status = 'pending';

# Time: 2024-01-15T14:18:45.901234Z
# User@Host: app_user[app_user] @ localhost []
# Query_time: 6.234567  Lock_time: 0.000189 Rows_sent: 456  Rows_examined: 125000
SET timestamp=1705327125;
SELECT * FROM users WHERE created_at >= '2023-01-01' AND status = 'active';

# Time: 2024-01-15T15:25:12.012345Z
# User@Host: app_user[app_user] @ localhost []
# Query_time: 3.678901  Lock_time: 0.000145 Rows_sent: 1  Rows_examined: 125000
SET timestamp=1705331112;
SELECT id, email FROM users WHERE email = 'charlie.brown@example.com';

# Time: 2024-01-15T16:08:37.123456Z
# User@Host: app_user[app_user] @ localhost []
# Query_time: 2.345678  Lock_time: 0.000056 Rows_sent: 78  Rows_examined: 125000
SET timestamp=1705333717;
SELECT * FROM users WHERE status = 'active' ORDER BY created_at;

# Time: 2024-01-15T16:52:23.234567Z
# User@Host: app_user[app_user] @ localhost []
# Query_time: 4.890123  Lock_time: 0.000167 Rows_sent: 203  Rows_examined: 125000
SET timestamp=1705336343;
SELECT username, created_at FROM users WHERE created_at > '2024-01-01' AND created_at < '2024-01-15';

# Time: 2024-01-15T17:33:48.345678Z
# User@Host: app_user[app_user] @ localhost []
# Query_time: 3.123456  Lock_time: 0.000099 Rows_sent: 1  Rows_examined: 125000
SET timestamp=1705338828;
SELECT * FROM users WHERE email = 'david.miller@example.com';

# Time: 2024-01-15T18:15:29.456789Z
# User@Host: app_user[app_user] @ localhost []
# Query_time: 2.678901  Lock_time: 0.000087 Rows_sent: 92  Rows_examined: 125000
SET timestamp=1705341329;
SELECT id, username, status FROM users WHERE status = 'suspended';