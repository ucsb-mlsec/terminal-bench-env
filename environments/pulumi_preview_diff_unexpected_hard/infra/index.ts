import * as pulumi from "@pulumi/pulumi";
import * as aws from "@pulumi/aws";
import * as random from "@pulumi/random";

// Get the current stack and project name
const stackName = pulumi.getStack();
const projectName = pulumi.getProject();

// Resource 1: webServer - Using timestamp in tags (non-deterministic)
const webServer = new aws.ec2.Instance("webServer", {
    instanceType: "t3.micro",
    ami: "ami-0c55b159cbfafe1f0",
    tags: {
        Name: "web-server",
        Environment: stackName,
        DeployedAt: new Date().toISOString(),
        Timestamp: Date.now().toString(),
        Project: projectName
    },
    userData: `#!/bin/bash
echo "Starting web server"
apt-get update
apt-get install -y nginx
systemctl start nginx
systemctl enable nginx
`,
    vpcSecurityGroupIds: ["sg-0123456789abcdef0"],
    subnetId: "subnet-0123456789abcdef0"
});

// Resource 2: database - Reading from environment variables without defaults
const database = new aws.rds.Instance("database", {
    allocatedStorage: 20,
    engine: "postgres",
    engineVersion: "13.7",
    instanceClass: "db.t3.micro",
    dbName: process.env.DB_NAME,
    username: process.env.DB_USERNAME,
    password: process.env.DB_PASSWORD,
    parameterGroupName: "default.postgres13",
    skipFinalSnapshot: true,
    tags: {
        Name: "primary-database",
        Environment: stackName,
        Owner: process.env.TEAM_OWNER
    },
    publiclyAccessible: false,
    vpcSecurityGroupIds: ["sg-0123456789abcdef1"],
    dbSubnetGroupName: "default"
});

// Resource 3: storageAccount - Using random number in properties
const storageAccount = new aws.s3.Bucket("storageAccount", {
    bucket: `storage-${Math.floor(Math.random() * 1000000)}`,
    acl: "private",
    tags: {
        Name: "storage-account",
        Environment: stackName,
        RandomId: Math.random().toString(36).substring(7)
    },
    versioning: {
        enabled: true
    },
    serverSideEncryptionConfiguration: {
        rule: {
            applyServerSideEncryptionByDefault: {
                sseAlgorithm: "AES256"
            }
        }
    }
});

// Resource 4: apiGateway - Missing ignoreChanges for computed property
const apiGateway = new aws.apigatewayv2.Api("apiGateway", {
    name: "main-api-gateway",
    protocolType: "HTTP",
    corsConfiguration: {
        allowOrigins: ["*"],
        allowMethods: ["GET", "POST", "PUT", "DELETE"],
        allowHeaders: ["Content-Type", "Authorization"]
    },
    tags: {
        Name: "api-gateway",
        Environment: stackName,
        ManagedBy: "Pulumi"
    },
    description: "Main API Gateway for the application"
    // Missing ignoreChanges for apiEndpoint which is computed
});

// Resource 5: staticBucket - Properly configured (should NOT cause issues)
const staticBucket = new aws.s3.Bucket("staticBucket", {
    bucket: `${projectName}-${stackName}-static-content`,
    acl: "public-read",
    tags: {
        Name: "static-content-bucket",
        Environment: stackName,
        Purpose: "static-assets"
    },
    website: {
        indexDocument: "index.html",
        errorDocument: "error.html"
    },
    versioning: {
        enabled: true
    }
}, {
    protect: false
});

// Additional properly configured resources
const securityGroup = new aws.ec2.SecurityGroup("webSecurityGroup", {
    name: `${projectName}-${stackName}-web-sg`,
    description: "Security group for web servers",
    vpcId: "vpc-0123456789abcdef0",
    ingress: [
        {
            protocol: "tcp",
            fromPort: 80,
            toPort: 80,
            cidrBlocks: ["0.0.0.0/0"]
        },
        {
            protocol: "tcp",
            fromPort: 443,
            toPort: 443,
            cidrBlocks: ["0.0.0.0/0"]
        }
    ],
    egress: [
        {
            protocol: "-1",
            fromPort: 0,
            toPort: 0,
            cidrBlocks: ["0.0.0.0/0"]
        }
    ],
    tags: {
        Name: `${projectName}-${stackName}-web-sg`,
        Environment: stackName
    }
});

// Export the important values
export const webServerPublicIp = webServer.publicIp;
export const webServerPublicDns = webServer.publicDns;
export const databaseEndpoint = database.endpoint;
export const storageAccountName = storageAccount.bucket;
export const apiGatewayEndpoint = apiGateway.apiEndpoint;
export const staticBucketWebsiteEndpoint = staticBucket.websiteEndpoint;
export const securityGroupId = securityGroup.id;