import * as pulumi from "@pulumi/pulumi";
import * as aws from "@pulumi/aws";

// Get stack configuration
const config = new pulumi.Config();
const environment = pulumi.getStack();

// Create an S3 bucket for data storage
// This bucket will be used by the Lambda function to process data
const dataBucket = new aws.s3.Bucket("data-bucket", {
    bucket: `my-data-bucket-${environment}`,
    acl: "private",
    
    // Server-side encryption configuration
    serverSideEncryptionConfiguration: {
        rule: {
            applyServerSideEncryptionByDefault: {
                sseAlgorithm: "AES256",
            },
        },
    },
    
    // Note: Versioning is not explicitly configured here
    // Tags are also not included in this declaration
});

// Create an IAM role for the Lambda function
// This role allows the Lambda to assume the role and access AWS services
const lambdaRole = new aws.iam.Role("lambda-role", {
    assumeRolePolicy: JSON.stringify({
        Version: "2012-10-17",
        Statement: [{
            Action: "sts:AssumeRole",
            Effect: "Allow",
            Principal: {
                Service: "lambda.amazonaws.com",
            },
        }],
    }),
});

// Attach the basic Lambda execution policy to the role
const lambdaPolicyAttachment = new aws.iam.RolePolicyAttachment("lambda-policy-attachment", {
    role: lambdaRole.name,
    policyArn: "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
});

// Attach a policy to allow the Lambda to access the S3 bucket
const s3Policy = new aws.iam.RolePolicy("lambda-s3-policy", {
    role: lambdaRole.id,
    policy: dataBucket.arn.apply(arn => JSON.stringify({
        Version: "2012-10-17",
        Statement: [{
            Effect: "Allow",
            Action: [
                "s3:GetObject",
                "s3:PutObject",
            ],
            Resource: `${arn}/*`,
        }],
    })),
});

// Create a Lambda function for data processing
// This function will process files uploaded to the S3 bucket
const processorLambda = new aws.lambda.Function("processor-lambda", {
    runtime: "nodejs14.x",
    handler: "index.handler",
    role: lambdaRole.arn,
    
    // Lambda code configuration
    code: new pulumi.asset.AssetArchive({
        ".": new pulumi.asset.FileArchive("./lambda"),
    }),
    
    // Memory allocation
    memorySize: 256,
    
    // Timeout in seconds
    timeout: 30,
    
    // Environment variables for the Lambda function
    environment: {
        variables: {
            BUCKET_NAME: dataBucket.bucket,
            ENVIRONMENT: environment,
        },
    },
    
    // Note: Description field is not included here
    // Reserved concurrent executions is also not configured
});

// Grant the S3 bucket permission to invoke the Lambda function
const bucketNotification = new aws.s3.BucketNotification("bucket-notification", {
    bucket: dataBucket.id,
    lambdaFunctions: [{
        lambdaFunctionArn: processorLambda.arn,
        events: ["s3:ObjectCreated:*"],
    }],
});

// Allow S3 to invoke the Lambda function
const lambdaPermission = new aws.lambda.Permission("lambda-permission", {
    action: "lambda:InvokeFunction",
    function: processorLambda.name,
    principal: "s3.amazonaws.com",
    sourceArn: dataBucket.arn,
});

// Export the bucket name and Lambda function ARN for reference
export const bucketName = dataBucket.bucket;
export const lambdaArn = processorLambda.arn;
export const lambdaName = processorLambda.name;