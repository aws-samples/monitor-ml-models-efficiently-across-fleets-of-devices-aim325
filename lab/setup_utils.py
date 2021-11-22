import sagemaker
from sagemaker import get_execution_role
import boto3
import botocore
from botocore.exceptions import ClientError
import json
import time

def setup_roles_and_policies(iot_device_role_name):
    role = get_execution_role()
    iam_client = boto3.client('iam')
    iam_resource = boto3.resource('iam')
    
    role_name = role.split('/')[-1]

    # use python sdk to attach a few more managed policy to sagemaker role
    policy_attach_res = iam_client.attach_role_policy(
        RoleName=role_name,
        PolicyArn="arn:aws:iam::aws:policy/AmazonEC2FullAccess"
    )

    policy_attach_res = iam_client.attach_role_policy(
        RoleName=role_name,
        PolicyArn="arn:aws:iam::aws:policy/service-role/AmazonEC2RoleforSSM"
    )

    policy_attach_res = iam_client.attach_role_policy(
        RoleName=role_name,
        PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
    )

    policy_attach_res = iam_client.attach_role_policy(
        RoleName=role_name,
        PolicyArn="arn:aws:iam::aws:policy/AmazonSSMFullAccess"
    )

    policy_attach_res = iam_client.attach_role_policy(
        RoleName=role_name,
        PolicyArn="arn:aws:iam::aws:policy/AWSGreengrassFullAccess"
    )

    ec2_role_name = "EdgeManager-Demo-EC2-" + str(time.time()).split(".")[0]

    trust_relationship_ec2_service = {
      "Version": "2012-10-17",
      "Statement": [
        {
          "Effect": "Allow",
          "Principal": {
            "Service": "ec2.amazonaws.com"
          },
          "Action": "sts:AssumeRole"
        }
      ]
    }

    # create EC2 role and its instance profile
    try:
        create_role_res = iam_client.create_role(
            RoleName=ec2_role_name,
            AssumeRolePolicyDocument=json.dumps(trust_relationship_ec2_service),
            Description='This is a EC2 role',
        )
    except ClientError as error:
        if error.response['Error']['Code'] == 'EntityAlreadyExists':
            print('Role already exists... hence exiting from here')
        else:
            print('Unexpected error occurred... Role could not be created', error)


    policy_attach_res = iam_client.attach_role_policy(
        RoleName=ec2_role_name,
        PolicyArn="arn:aws:iam::aws:policy/AmazonS3FullAccess"
    )

    policy_attach_res = iam_client.attach_role_policy(
        RoleName=ec2_role_name,
        PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
    )

    policy_attach_res = iam_client.attach_role_policy(
        RoleName=ec2_role_name,
        PolicyArn="arn:aws:iam::aws:policy/CloudWatchAgentAdminPolicy"
    )

    policy_attach_res = iam_client.attach_role_policy(
        RoleName=ec2_role_name,
        PolicyArn="arn:aws:iam::aws:policy/CloudWatchAgentAdminPolicy"
    )

    account_id = role.split(":")[4]

    # Create a policy
    my_managed_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "CreateTokenExchangeRole",
                "Effect": "Allow",
                "Action": [
                    "iam:AttachRolePolicy",
                    "iam:CreatePolicy",
                    "iam:CreateRole",
                    "iam:GetPolicy",
                    "iam:GetRole",
                    "iam:PassRole"
                ],
                "Resource": [
                    f"arn:aws:iam::{account_id}:role/{iot_device_role_name}",
                    f"arn:aws:iam::{account_id}:policy/{iot_device_role_name}Access",
                    f"arn:aws:iam::aws:policy/{iot_device_role_name}Access"
                ]
            },
            {
                "Effect": "Allow",
                "Action": [
                    "iot:AddThingToThingGroup",
                    "iot:AttachPolicy",
                    "iot:AttachThingPrincipal",
                    "iot:CreateKeysAndCertificate",
                    "iot:CreatePolicy",
                    "iot:CreateRoleAlias",
                    "iot:CreateThing",
                    "iot:CreateThingGroup",
                    "iot:DescribeEndpoint",
                    "iot:DescribeRoleAlias",
                    "iot:DescribeThingGroup",
                    "iot:GetPolicy",
                    "sts:GetCallerIdentity"
                ],
                "Resource": "*"
            },
            {
                "Sid": "DeployDevTools",
                "Effect": "Allow",
                "Action": [
                    "greengrass:CreateDeployment",
                    "iot:CancelJob",
                    "iot:CreateJob",
                    "iot:DeleteThingShadow",
                    "iot:DescribeJob",
                    "iot:DescribeThing",
                    "iot:DescribeThingGroup",
                    "iot:GetThingShadow",
                    "iot:UpdateJob",
                    "iot:UpdateThingShadow"
                ],
                "Resource": "*"
            }
        ]
    }

    response = iam_client.create_policy(
      PolicyName='ggv2_provision_policy' + str(time.time()).split(".")[0],
      PolicyDocument=json.dumps(my_managed_policy)
    )


    policy_attach_res = iam_client.attach_role_policy(
        RoleName=ec2_role_name,
        PolicyArn=response['Policy']['Arn']
    )


    pass_ec2_role_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "iam:PassRole",
                "Resource": f"arn:aws:iam::{account_id}:role/{ec2_role_name}"
            }
        ]
    }

    response = iam_client.create_policy(
      PolicyName='pass_ec2_role_policy' + str(time.time()).split(".")[0],
      PolicyDocument=json.dumps(pass_ec2_role_policy)
    )

    response = iam_client.create_instance_profile(
        InstanceProfileName=ec2_role_name
    )

    instance_profile = iam_resource.InstanceProfile(
        ec2_role_name
    )

    instance_profile.add_role(
        RoleName=ec2_role_name
    )
    
    ## wait for 10 secs until the instance profile was created fully
    time.sleep(10)
    
    return ec2_role_name

    
def modify_device_role(iot_device_role_name):
    iam_client = boto3.client('iam')
    
    # Create a policy
    download_component_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject"
                ],
                "Resource": [
                    "arn:aws:s3:::*SageMaker*",
                    "arn:aws:s3:::*Sagemaker*",
                    "arn:aws:s3:::*sagemaker*"
                ]
            }
        ]
    }

    response = iam_client.create_policy(
      PolicyName='download_component_policy' + str(time.time()).split(".")[0],
      PolicyDocument=json.dumps(download_component_policy)
    )

    policy_attach_res = iam_client.attach_role_policy(
        RoleName=iot_device_role_name,
        PolicyArn=response['Policy']['Arn']
    )

    policy_attach_res = iam_client.attach_role_policy(
        RoleName=iot_device_role_name,
        PolicyArn="arn:aws:iam::aws:policy/service-role/AmazonSageMakerEdgeDeviceFleetPolicy"
    )

    response = iam_client.update_assume_role_policy(
        PolicyDocument='''{
            "Version": "2012-10-17",
            "Statement": [
               {
                 "Effect": "Allow",
                 "Principal": {"Service": "credentials.iot.amazonaws.com"},
                 "Action": "sts:AssumeRole"
               },
               {
                 "Effect": "Allow",
                 "Principal": {"Service": "sagemaker.amazonaws.com"},
                 "Action": "sts:AssumeRole"
               }
            ]
        }''',
        RoleName=iot_device_role_name,
    )
   
    ## wait for 30 secs until IAM changes fully propogate
    time.sleep(30)
    
    account_id = get_execution_role().split(":")[4]
    role_arn = f"arn:aws:iam::{account_id}:role/{iot_device_role_name}"
    return role_arn
