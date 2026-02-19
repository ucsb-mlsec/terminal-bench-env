import * as pulumi from "@pulumi/pulumi";
import * as aws from "@pulumi/aws";

// Configuration helper functions
export function getConfig(key: string, defaultValue?: string): string {
    const config = new pulumi.Config();
    return config.get(key) || defaultValue || "";
}

export function requireConfig(key: string): string {
    const config = new pulumi.Config();
    return config.require(key);
}

export function getConfigNumber(key: string, defaultValue: number): number {
    const config = new pulumi.Config();
    return config.getNumber(key) || defaultValue;
}

// Get environment name
const environment = getConfig("environment", "dev");
const projectName = getConfig("project", "myapp");

// Lambda Function with timestamp in environment variables
export const lambdaFunction = new aws.lambda.Function("lambdaFunction", {
    runtime: aws.lambda.Runtime.NodeJS18dX,
    code: new pulumi.asset.AssetArchive({
        ".": new pulumi.asset.FileArchive("./lambda"),
    }),
    handler: "index.handler",
    role: new aws.iam.Role("lambdaRole", {
        assumeRolePolicy: JSON.stringify({
            Version: "2012-10-17",
            Statement: [{
                Action: "sts:AssumeRole",
                Principal: {
                    Service: "lambda.amazonaws.com",
                },
                Effect: "Allow",
            }],
        }),
    }).arn,
    environment: {
        variables: {
            ENVIRONMENT: environment,
            PROJECT_NAME: projectName,
            DEPLOYED_AT: new Date().toISOString(),
            LOG_LEVEL: "info",
        },
    },
    description: `Lambda function deployed at ${new Date().toISOString()}`,
    timeout: 30,
    memorySize: 512,
});

// Queue Service with random capacity settings
export const queueService = new aws.sqs.Queue("queueService", {
    name: `${projectName}-${environment}-queue`,
    visibilityTimeoutSeconds: 300,
    messageRetentionSeconds: 345600,
    maxMessageSize: 262144,
    delaySeconds: Math.floor(Math.random() * 10),
    receiveWaitTimeSeconds: Math.floor(Math.random() * 20) + 1,
    tags: {
        Environment: environment,
        Project: projectName,
        ManagedBy: "pulumi",
    },
});

// Export resource ARNs for use in other files
export const lambdaArn = lambdaFunction.arn;
export const queueUrl = queueService.url;
export const queueArn = queueService.arn;