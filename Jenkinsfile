pipeline{
    agent any

    stages{
        stage('Build'){
            steps{
                echo 'Building the application...'
                checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: '29081395-60db-4b6c-ac89-07b53967d992', url: 'https://github.com/csergey88/hotel-reservation']])
            }
        }
        stage('Test'){
            steps{
                echo 'Testing the application...'
            }
        }
        stage('Deploy'){
            steps{
                echo 'Deploying the application...'
            }
        }
    }
}