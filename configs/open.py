#!/usr/bin/python3

import argparse
import os
import socket
import subprocess
import sys


dbConn = {}


def executeCommand(cmd, input=None, cwd=None, env=None, rstrp=True, timeout=None):
    print("Cmd: " + cmd)

    pipe = subprocess.Popen(
        cmd,
        shell=True,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.PIPE if input is not None else None
    )

    try:
        (output, errout) = pipe.communicate(input=input, timeout=timeout)
    except subprocess.TimeoutExpired:
        pipe.kill()
        (output, errout) = pipe.communicate()

        result = output.decode("utf8", errors="replace")
        print("ERROR: Command timed out: " + cmd)

        if result.strip() != "":
            print(result)

        sys.exit(1)

    status = pipe.returncode

    result = output.decode("utf8", errors="replace")
    if rstrp:
        result = result.rstrip()

    if status != 0:
        print("ERROR: Failed to execute the command: " + cmd)

        if result.strip() != "":
            print(result)

        sys.exit(1)

    return (status, result)


def sqlEscape(value):
    if value is None:
        return ""

    return str(value).replace("\\", "\\\\").replace("'", "''")


def deriveUserIdFromParty(partyId):
    partyId = str(partyId).strip().lower()

    if partyId == "":
        return ""

    firstPart = partyId.split()[0]

    if firstPart.startswith("cls"):
        prefix = firstPart
    elif firstPart.startswith("cl"):
        prefix = "cls" + firstPart[2:]
    else:
        prefix = firstPart

    return prefix + "_tradeapi"


def normalizeApiUserid(partyId, apiUserid):
    if apiUserid is None or str(apiUserid).strip() == "":
        return deriveUserIdFromParty(partyId)

    apiUserid = str(apiUserid).strip().lower()

    if apiUserid.startswith("cl") and not apiUserid.startswith("cls"):
        apiUserid = "cls" + apiUserid[2:]

    return apiUserid


def executeMysqlScalar(sql, mysqlBaseCmd, mysqlEnv, targetDb, timeout=30):
    cmd = mysqlBaseCmd + "--database=" + targetDb + " -N -B "

    (status, result) = executeCommand(
        cmd,
        input=(sql.strip() + "\n").encode("utf8"),
        env=mysqlEnv,
        timeout=timeout
    )

    lines = [line.strip() for line in result.splitlines() if line.strip()]

    if len(lines) == 0:
        return ""

    value = lines[0]

    if value.upper() == "NULL":
        return ""

    return value


def executeMysqlStep(stepName, sql, mysqlBaseCmd, mysqlEnv, targetDb, timeout=60):
    print("")
    print("========================================")
    print("RUNNING SQL STEP: " + stepName)
    print("========================================")

    cmd = mysqlBaseCmd + "--database=" + targetDb + " "

    sqlWithDebug = (
        "SELECT CONCAT('RUNNING_IN_DB=', DATABASE()) AS selected_db;\n"
        + sql.strip()
        + "\n"
    )

    (status, result) = executeCommand(
        cmd,
        input=sqlWithDebug.encode("utf8"),
        env=mysqlEnv,
        rstrp=False,
        timeout=timeout
    )

    if result.strip() != "":
        print(result)
    else:
        print("No output returned for SQL step: " + stepName)

    print("COMPLETED SQL STEP: " + stepName)

    return (status, result)


def executeMysqlSteps(stepList, mysqlBaseCmd, mysqlEnv, targetDb):
    for step in stepList:
        stepName = step[0]
        stepSql = step[1]
        executeMysqlStep(stepName, stepSql, mysqlBaseCmd, mysqlEnv, targetDb)


def getEnvState(mysqlBaseCmd, mysqlEnv, targetDb):
    uriCount = executeMysqlScalar(
        """
SELECT COUNT(*)
FROM urimaster
WHERE uri_pattern = '/ws/trade-input/*';
""",
        mysqlBaseCmd,
        mysqlEnv,
        targetDb
    )

    if int(uriCount) > 1:
        print("ERROR: Multiple urimaster records found for /ws/trade-input/*")
        sys.exit(1)

    uriId = ""
    if int(uriCount) == 1:
        uriId = executeMysqlScalar(
            """
SELECT MAX(uri_id)
FROM urimaster
WHERE uri_pattern = '/ws/trade-input/*';
""",
            mysqlBaseCmd,
            mysqlEnv,
            targetDb
        )

    groupCount = executeMysqlScalar(
        """
SELECT COUNT(*)
FROM groupmaster
WHERE group_name = 'cls_trade_api'
   OR group_code = 'cls_trade_api';
""",
        mysqlBaseCmd,
        mysqlEnv,
        targetDb
    )

    if int(groupCount) > 1:
        print("ERROR: Multiple groupmaster records found for cls_trade_api")
        sys.exit(1)

    groupId = ""
    if int(groupCount) == 1:
        groupId = executeMysqlScalar(
            """
SELECT MAX(group_id)
FROM groupmaster
WHERE group_name = 'cls_trade_api'
   OR group_code = 'cls_trade_api';
""",
            mysqlBaseCmd,
            mysqlEnv,
            targetDb
        )

    groupUriCount = "0"
    groupUriId = ""

    if uriId != "" and groupId != "":
        groupUriCount = executeMysqlScalar(
            """
SELECT COUNT(*)
FROM group_uri_xref
WHERE group_id = %s
  AND uri_id = %s;
""" % (groupId, uriId),
            mysqlBaseCmd,
            mysqlEnv,
            targetDb
        )

        if int(groupUriCount) > 1:
            print("ERROR: Multiple group_uri_xref records found for group_id/uri_id")
            sys.exit(1)

        if int(groupUriCount) == 1:
            groupUriId = executeMysqlScalar(
                """
SELECT MAX(group_uri_id)
FROM group_uri_xref
WHERE group_id = %s
  AND uri_id = %s;
""" % (groupId, uriId),
                mysqlBaseCmd,
                mysqlEnv,
                targetDb
            )

    print("")
    print("ENV STATE")
    print("uri_count: " + str(uriCount))
    print("uri_id: " + str(uriId))
    print("group_count: " + str(groupCount))
    print("group_id: " + str(groupId))
    print("group_uri_count: " + str(groupUriCount))
    print("group_uri_id: " + str(groupUriId))

    return {
        "uri_count": uriCount,
        "uri_id": uriId,
        "group_count": groupCount,
        "group_id": groupId,
        "group_uri_count": groupUriCount,
        "group_uri_id": groupUriId
    }


def buildEnvSqlSteps(envState):
    steps = []

    uriId = envState["uri_id"]
    groupId = envState["group_id"]
    groupUriId = envState["group_uri_id"]

    if uriId == "":
        steps.append((
            "env_01_insert_urimaster",
            """
SELECT 'STARTING ENV STEP 1: URIMASTER' AS status;

SELECT @uri_id := COALESCE(MAX(uri_id), 0) + 1
FROM urimaster;

INSERT INTO urimaster
(
    uri_id,
    uri_pattern,
    uri_desc,
    uri_auth_type,
    active,
    created_by,
    created_date,
    modified_by,
    modified_date,
    http_method
)
VALUES
(
    @uri_id,
    '/ws/trade-input/*',
    'Trade input API',
    'group',
    1,
    NULL,
    NOW(),
    NULL,
    NOW(),
    'ALL'
);

SELECT 'VERIFY URIMASTER' AS status;
SELECT * FROM urimaster WHERE uri_id = @uri_id;
"""
        ))
    else:
        print("Skipping urimaster insert. Existing uri_id: " + uriId)

    if groupId == "":
        steps.append((
            "env_02_insert_groupmaster",
            """
SELECT 'STARTING ENV STEP 2: GROUPMASTER' AS status;

SELECT @group_id := COALESCE(MAX(group_id), 0) + 1
FROM groupmaster;

INSERT INTO groupmaster
(
    group_id,
    app_id,
    group_name,
    group_code,
    group_desc,
    group_type,
    ui_display,
    active,
    created_by,
    created_date,
    modified_by,
    modified_date,
    mfa_level
)
VALUES
(
    @group_id,
    10,
    'cls_trade_api',
    'cls_trade_api',
    'cls_trade_api',
    NULL,
    1,
    1,
    NULL,
    NOW(),
    NULL,
    NOW(),
    0
);

SELECT 'VERIFY GROUPMASTER' AS status;
SELECT * FROM groupmaster WHERE group_id = @group_id;
"""
        ))
    else:
        print("Skipping groupmaster insert. Existing group_id: " + groupId)

    if groupUriId == "":
        steps.append((
            "env_03_insert_group_uri_xref",
            """
SELECT 'STARTING ENV STEP 3: GROUP_URI_XREF' AS status;

SELECT @uri_id := MAX(uri_id)
FROM urimaster
WHERE uri_pattern = '/ws/trade-input/*';

SELECT @group_id := MAX(group_id)
FROM groupmaster
WHERE group_name = 'cls_trade_api'
   OR group_code = 'cls_trade_api';

SELECT @group_uri_id := COALESCE(MAX(group_uri_id), 0) + 1
FROM group_uri_xref;

INSERT INTO group_uri_xref
(
    group_uri_id,
    group_id,
    uri_id,
    created_by,
    created_date,
    modified_by,
    modified_date
)
VALUES
(
    @group_uri_id,
    @group_id,
    @uri_id,
    NULL,
    NOW(),
    NULL,
    NULL
);

SELECT 'VERIFY GROUP_URI_XREF' AS status;
SELECT * FROM group_uri_xref WHERE group_uri_id = @group_uri_id;
"""
        ))
    else:
        print("Skipping group_uri_xref insert. Existing group_uri_id: " + groupUriId)

    return steps


def getUserState(partyId, apiUserid, mysqlBaseCmd, mysqlEnv, targetDb):
    partyId = sqlEscape(partyId)
    apiUserid = sqlEscape(apiUserid)

    orgCount = executeMysqlScalar(
        """
SELECT COUNT(*)
FROM organizationmaster
WHERE party_id = '%s';
""" % partyId,
        mysqlBaseCmd,
        mysqlEnv,
        targetDb
    )

    if int(orgCount) == 0:
        print("ERROR: No organization found for party_id: " + partyId)
        sys.exit(1)

    if int(orgCount) > 1:
        print("ERROR: Multiple organizations found for party_id: " + partyId)
        sys.exit(1)

    organizationId = executeMysqlScalar(
        """
SELECT MAX(organization_id)
FROM organizationmaster
WHERE party_id = '%s';
""" % partyId,
        mysqlBaseCmd,
        mysqlEnv,
        targetDb
    )

    groupCount = executeMysqlScalar(
        """
SELECT COUNT(*)
FROM groupmaster
WHERE group_name = 'cls_trade_api'
   OR group_code = 'cls_trade_api';
""",
        mysqlBaseCmd,
        mysqlEnv,
        targetDb
    )

    if int(groupCount) == 0:
        print("ERROR: ENV onboarding missing. Could not find cls_trade_api group.")
        sys.exit(1)

    if int(groupCount) > 1:
        print("ERROR: Multiple groupmaster records found for cls_trade_api")
        sys.exit(1)

    groupId = executeMysqlScalar(
        """
SELECT MAX(group_id)
FROM groupmaster
WHERE group_name = 'cls_trade_api'
   OR group_code = 'cls_trade_api';
""",
        mysqlBaseCmd,
        mysqlEnv,
        targetDb
    )

    userCount = executeMysqlScalar(
        """
SELECT COUNT(*)
FROM usermaster
WHERE login = '%s@cls-services.com';
""" % apiUserid,
        mysqlBaseCmd,
        mysqlEnv,
        targetDb
    )

    if int(userCount) > 1:
        print("ERROR: Multiple usermaster records found for login: " + apiUserid + "@cls-services.com")
        sys.exit(1)

    userId = ""
    if int(userCount) == 1:
        userId = executeMysqlScalar(
            """
SELECT MAX(id)
FROM usermaster
WHERE login = '%s@cls-services.com';
""" % apiUserid,
            mysqlBaseCmd,
            mysqlEnv,
            targetDb
        )

    groupUserCount = "0"
    groupUserId = ""

    if userId != "" and groupId != "":
        groupUserCount = executeMysqlScalar(
            """
SELECT COUNT(*)
FROM group_user_xref
WHERE group_id = %s
  AND user_id = '%s';
""" % (groupId, sqlEscape(userId)),
            mysqlBaseCmd,
            mysqlEnv,
            targetDb
        )

        if int(groupUserCount) > 1:
            print("ERROR: Multiple group_user_xref records found for user_id/group_id")
            sys.exit(1)

        if int(groupUserCount) == 1:
            groupUserId = executeMysqlScalar(
                """
SELECT MAX(group_user_id)
FROM group_user_xref
WHERE group_id = %s
  AND user_id = '%s';
""" % (groupId, sqlEscape(userId)),
                mysqlBaseCmd,
                mysqlEnv,
                targetDb
            )

    print("")
    print("USER STATE")
    print("organization_count: " + str(orgCount))
    print("organization_id: " + str(organizationId))
    print("group_count: " + str(groupCount))
    print("group_id: " + str(groupId))
    print("user_count: " + str(userCount))
    print("user_id: " + str(userId))
    print("group_user_count: " + str(groupUserCount))
    print("group_user_id: " + str(groupUserId))

    return {
        "organization_count": orgCount,
        "organization_id": organizationId,
        "group_count": groupCount,
        "group_id": groupId,
        "user_count": userCount,
        "user_id": userId,
        "group_user_count": groupUserCount,
        "group_user_id": groupUserId
    }


def buildUserSqlSteps(partyId, apiUserid, username, userState):
    partyId = sqlEscape(partyId)
    apiUserid = sqlEscape(apiUserid)
    username = sqlEscape(username)

    organizationId = userState["organization_id"]
    groupId = userState["group_id"]
    userId = userState["user_id"]
    groupUserId = userState["group_user_id"]

    steps = []

    if organizationId == "":
        print("ERROR: No organization found for party_id: " + partyId)
        print("User onboarding cannot continue without organization_id.")
        sys.exit(1)

    if groupId == "":
        print("ERROR: ENV onboarding missing. Could not find cls_trade_api group.")
        print("Run env onboarding first or run with --env --user.")
        sys.exit(1)

    if userId == "":
        steps.append((
            "user_01_insert_usermaster",
            """
SELECT 'STARTING USER STEP 1: USERMASTER' AS status;

SELECT @organization_id := MAX(organization_id)
FROM organizationmaster
WHERE party_id = '%s';

SELECT 'ORG LOOKUP' AS status, @organization_id AS organization_id;

SELECT @random_uuid := UUID();

INSERT INTO usermaster
(
    id,
    login,
    display_name,
    first_name,
    last_name,
    middle_name,
    honorific_prefix,
    honorific_suffix,
    email,
    mobile_phone,
    user_type,
    department,
    role,
    status,
    modified_by,
    allow_multi_session,
    organization_id,
    enforce_internal_network
)
VALUES
(
    @random_uuid,
    '%s@cls-services.com',
    '%s',
    '%s',
    '%s',
    NULL,
    NULL,
    NULL,
    'noreply@cls-bank.com',
    NULL,
    'API',
    'Operations',
    NULL,
    1,
    NULL,
    0,
    @organization_id,
    0
);

SELECT 'VERIFY USERMASTER' AS status;
SELECT * FROM usermaster WHERE id = @random_uuid;
""" % (
                partyId,
                apiUserid,
                username,
                username,
                username
            )
        ))
    else:
        print("Skipping usermaster insert. Existing user_id: " + userId)

    if groupUserId == "":
        steps.append((
            "user_02_insert_group_user_xref",
            """
SELECT 'STARTING USER STEP 2: GROUP_USER_XREF' AS status;

SELECT @user_id := MAX(id)
FROM usermaster
WHERE login = '%s@cls-services.com';

SELECT @existing_group_id := MAX(group_id)
FROM groupmaster
WHERE group_name = 'cls_trade_api'
   OR group_code = 'cls_trade_api';

SELECT @group_user_id := COALESCE(MAX(group_user_id), 0) + 1
FROM group_user_xref;

INSERT INTO group_user_xref
(
    group_user_id,
    group_id,
    user_id,
    created_by,
    created_date,
    modified_by,
    modified_date
)
VALUES
(
    @group_user_id,
    @existing_group_id,
    @user_id,
    NULL,
    NOW(),
    NULL,
    NULL
);

SELECT 'VERIFY GROUP_USER_XREF' AS status;
SELECT * FROM group_user_xref WHERE group_user_id = @group_user_id;
""" % apiUserid
        ))
    else:
        print("Skipping group_user_xref insert. Existing group_user_id: " + groupUserId)

    return steps


def setupDBConnection(runEnv, runUser, partyId, apiUserid):
    global dbConn

    try:
        fqdn = socket.gethostname().strip().lower()
        print("hostname: " + fqdn)

        if "cit" in fqdn:
            env_root = "cit"
        elif "dev" in fqdn:
            env_root = "dev"
        else:
            print("ERROR: Unable to determine env_root from hostname")
            sys.exit(1)

        print("env_root: " + env_root)

        nslookupHost = "rds-aurora-mysql." + env_root + ".clsnet"
        nslookupCmd = (
            "nslookup " + nslookupHost +
            " | grep canonical | head -1 | sed 's/.$//' | awk '{print $5}'"
        )
        (status, rdsHost) = executeCommand(nslookupCmd, timeout=30)

        rdsHost = rdsHost.strip()
        if rdsHost == "":
            print("ERROR: Failed to resolve RDS host from: " + nslookupHost)
            sys.exit(1)

        print("RDSHOST: " + rdsHost)

        dbUser = "aducpnet"
        dbPort = "3306"
        sslCa = "/cls/appl/env/truststore/eu-west-2-bundle_RSA2048.pem"

        tokenCmd = (
            "aws rds generate-db-auth-token "
            "--hostname " + rdsHost + " "
            "--port " + dbPort + " "
            "--username " + dbUser
        )
        (status, token) = executeCommand(tokenCmd, timeout=30)

        token = token.strip()
        if token == "":
            print("ERROR: Failed to generate DB auth token")
            sys.exit(1)

        mysqlEnv = os.environ.copy()
        mysqlEnv["MYSQL_PWD"] = token

        mysqlBaseCmd = (
            "mysql "
            "--host=" + rdsHost + " "
            "--port=" + dbPort + " "
            "--ssl-ca=" + sslCa + " "
            "--user=" + dbUser + " "
            "--enable-cleartext-plugin "
            "--batch "
            "--raw "
        )

        findDbCmd = mysqlBaseCmd + '-N -B -e "show databases;"'
        (status, result) = executeCommand(findDbCmd, env=mysqlEnv, timeout=30)

        dbList = [line.strip() for line in result.splitlines() if line.strip()]
        ucpDbs = [db for db in dbList if "ucp" in db.lower()]

        if len(ucpDbs) == 0:
            print("ERROR: No DB found with 'ucp' in it")
            sys.exit(1)

        exactMatches = [db for db in ucpDbs if db.lower() == "ucp"]
        if len(exactMatches) > 0:
            targetDb = exactMatches[0]
        else:
            targetDb = ucpDbs[0]

        print("Selected DB: " + targetDb)

        dbConn = {
            "hostname": fqdn,
            "env_root": env_root,
            "rds_host": rdsHost,
            "db_user": dbUser,
            "db_port": dbPort,
            "ssl_ca": sslCa,
            "mysql_env": mysqlEnv,
            "mysql_base_cmd": mysqlBaseCmd,
            "target_db": targetDb
        }

        testCmd = mysqlBaseCmd + "--database=" + targetDb + " -N -B -e \"SELECT CONCAT('SELECTED_DB=', DATABASE());\""
        (status, testResult) = executeCommand(testCmd, env=mysqlEnv, timeout=30)

        print("DB SMOKE TEST OUTPUT:")
        print(testResult)

        if "SELECTED_DB=" + targetDb not in testResult:
            print("ERROR: MySQL did not select expected DB: " + targetDb)
            sys.exit(1)

        if not runEnv and not runUser:
            print("ERROR: Nothing to run. Use --env and/or --user")
            sys.exit(1)

        if runEnv:
            envState = getEnvState(mysqlBaseCmd, mysqlEnv, targetDb)
            envSteps = buildEnvSqlSteps(envState)

            if len(envSteps) == 0:
                print("Env onboarding already complete. Nothing to run.")
            else:
                executeMysqlSteps(envSteps, mysqlBaseCmd, mysqlEnv, targetDb)

                print("")
                print("Re-checking env onboarding after insert...")
                finalEnvState = getEnvState(mysqlBaseCmd, mysqlEnv, targetDb)

                if (
                    finalEnvState["uri_id"] == "" or
                    finalEnvState["group_id"] == "" or
                    finalEnvState["group_uri_id"] == ""
                ):
                    print("ERROR: Env onboarding incomplete after running SQL.")
                    sys.exit(1)

                print("Env onboarding completed successfully.")

        if runUser:
            if partyId is None or str(partyId).strip() == "":
                print("ERROR: --party is required when --user is set")
                sys.exit(1)

            apiUserid = normalizeApiUserid(partyId, apiUserid)
            username = apiUserid

            print("USERID: " + apiUserid)
            print("USERNAME: " + username)

            userState = getUserState(partyId, apiUserid, mysqlBaseCmd, mysqlEnv, targetDb)
            userSteps = buildUserSqlSteps(partyId, apiUserid, username, userState)

            if len(userSteps) == 0:
                print("User onboarding already complete. Nothing to run.")
            else:
                executeMysqlSteps(userSteps, mysqlBaseCmd, mysqlEnv, targetDb)

                print("")
                print("Re-checking user onboarding after insert...")
                finalUserState = getUserState(partyId, apiUserid, mysqlBaseCmd, mysqlEnv, targetDb)

                if (
                    finalUserState["organization_id"] == "" or
                    finalUserState["group_id"] == "" or
                    finalUserState["user_id"] == "" or
                    finalUserState["group_user_id"] == ""
                ):
                    print("ERROR: User onboarding incomplete after running SQL.")
                    sys.exit(1)

                print("User onboarding completed successfully.")

    except Exception as e:
        print("ERROR: setupDBConnection failed: " + str(e))
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Trade API onboarding")

    parser.add_argument(
        "--env",
        action="store_true",
        help="run environment-only onboarding"
    )

    parser.add_argument(
        "--user",
        action="store_true",
        help="run user-only onboarding"
    )

    parser.add_argument(
        "--party",
        dest="party_id",
        help="party id, for example: CL52 OBO"
    )

    parser.add_argument(
        "--userid",
        dest="api_userid",
        help="api userid, for example: cls52_tradeapi"
    )

    pa = parser.parse_args()

    if not pa.env and not pa.user:
        print("ERROR: Choose at least one onboarding mode: --env and/or --user")
        print("")
        print("Examples:")
        print("  python3 tradeapi-onboarding.py --env")
        print("  python3 tradeapi-onboarding.py --user --party \"CL52 OBO\"")
        print("  python3 tradeapi-onboarding.py --user --party \"CL52 OBO\" --userid \"cls52_tradeapi\"")
        print("  python3 tradeapi-onboarding.py --env --user --party \"CL52 OBO\"")
        sys.exit(1)

    if pa.user:
        if pa.party_id is None or str(pa.party_id).strip() == "":
            print("ERROR: --party is required when --user is set")
            sys.exit(1)

    if not pa.user:
        if pa.party_id is not None or pa.api_userid is not None:
            print("ERROR: --party and --userid should only be used with --user")
            sys.exit(1)

    setupDBConnection(pa.env, pa.user, pa.party_id, pa.api_userid)


if __name__ == "__main__":
    main()