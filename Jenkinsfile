// Same stages as .github/workflows/ci.yml. Runs on the local Jenkins in ci/jenkins/.
pipeline {
    agent any

    options {
        timestamps()
        timeout(time: 30, unit: 'MINUTES')
        disableConcurrentBuilds()
    }

    environment {
        COMPOSE_PROJECT_NAME = "smishing-ci-${env.BUILD_NUMBER}"
        MODEL_PORT = '18000'
        GATEWAY_PORT = '18080'
    }

    stages {
        stage('Python: lint and test') {
            parallel {
                stage('ml') {
                    steps {
                        dir('ml') {
                            sh 'uv sync --frozen'
                            sh 'uv run ruff check . && uv run ruff format --check .'
                            sh 'uv run pytest -q'
                        }
                    }
                }
                stage('model-service') {
                    steps {
                        dir('model-service') {
                            sh 'uv sync --frozen'
                            sh 'uv run ruff check . && uv run ruff format --check .'
                            sh 'uv run pytest -q'
                        }
                    }
                }
            }
        }

        stage('Go: vet and test') {
            steps {
                sh 'go vet ./...'
                sh 'go test ./...'
            }
        }

        stage('Fetch released model') {
            steps {
                sh 'sh scripts/fetch-model.sh'
            }
        }

        stage('Regression gate') {
            steps {
                sh 'docker compose up -d --build --wait model-service gateway'
                sh 'docker compose run --rm --build evalgate'
            }
        }
    }

    post {
        failure {
            sh 'docker compose logs'
        }
        always {
            sh 'docker compose down --remove-orphans'
        }
    }
}
