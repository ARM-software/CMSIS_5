@Library("cmsis")

DOCKERINFO = [
    'linux_staging': [
        'registryUrl': 'mcu--docker-staging.eu-west-1.artifactory.aws.arm.com',
        'registryCredentialsId': 'artifactory',
        'k8sPullSecret': 'artifactory-mcu-docker-staging',
        'namespace': 'mcu--docker-staging',
        'image': 'cmsis_fusa/linux',
        'label': "${JENKINS_ENV}-${JOB_BASE_NAME}-${BUILD_NUMBER}"
    ],
    'linux_production': [
        'registryUrl': 'mcu--docker.eu-west-1.artifactory.aws.arm.com',
        'registryCredentialsId': 'artifactory',
        'namespace': 'mcu--docker',
        'k8sPullSecret': 'artifactory-mcu-docker',
        'image': 'cmsis_fusa/linux',
        'label': 'aws'
    ],
    'windows_staging': [
        'registryUrl': 'mcu--docker-staging.eu-west-1.artifactory.aws.arm.com',
        'registryCredentialsId': 'artifactory',
        'namespace': 'mcu--docker-staging',
        'image': 'cmsis_fusa/windows',
        'label': "${JENKINS_ENV}-${JOB_BASE_NAME}-${BUILD_NUMBER}"
    ],
    'windows_production': [
        'registryUrl': 'mcu--docker.eu-west-1.artifactory.aws.arm.com',
        'registryCredentialsId': 'artifactory',
        'namespace': 'mcu--docker',
        'image': 'cmsis_fusa/windows',
        'label': 'aws'
    ]
]

dockerinfo_linux = DOCKERINFO['linux_production']
dockerinfo_windows = DOCKERINFO['windows_production']

isPrecommit = (JOB_BASE_NAME == 'pre_commit')
isNightly = (JOB_BASE_NAME == 'nightly')

patternCoreM = [
    '^CMSIS/Core/Include/.*',
    '^Device/ARM/ARMCM.*'
]

patternCoreA = [
    '^CMSIS/Core_A/Include/.*',
    '^Device/ARM/ARMCA.*'
]

patternCoreValidation = [
    '^CMSIS/CoreValidation/.*'
]

CONFIGURATIONS = [
    'pre_commit': [
        'mdevices': ['CM0', 'CM3', 'CM4FP', 'CM7DP', 'CM23', 'CM33NS', 'CM35PS'],
        'adevices': ['CA7', 'CA9neon'],
        'devices' : [],
        'configs' : [
            'AC6': ['low', 'tiny'],
            'AC6LTM': ['low', 'tiny']
        ]
    ],
    'nightly':[
        'devices' : ['CM0', 'CM0plus', 'CM3', 'CM4', 'CM4FP', 'CM7', 'CM7SP', 'CM7DP',
                     'CM23', 'CM23S', 'CM23NS', 'CM33', 'CM33S', 'CM33NS',
                     'CM35P', 'CM35PS', 'CM35PNS',
                     'CA5', 'CA5neon', 'CA7', 'CA7neon', 'CA9', 'CA9neon'],
        'configs' : [
            'AC6': ['low', 'mid', 'high', 'size', 'tiny'],
            'AC6LTM': ['low', 'mid', 'high', 'size', 'tiny']
        ]
    ]
]
CONFIGURATION = CONFIGURATIONS[JOB_BASE_NAME]

// ---- PIPELINE CODE ----

def getChangeset() {
    def fileset = sh encoding: 'UTF-8', label: '', returnStdout: true, script: 'git diff --name-only HEAD~1..HEAD'
    return fileset.split('\n')
}

def fileSetMatches(fileset, patternset) {
    return patternset.any { p ->
        fileset.any{ f -> f ==~ p }
    }
}

FORCE_BUILD = false
CORE_VALIDATION = true
COMMIT = null
VERSION = null

pipeline {
    options {
        timestamps()
        timeout(time: 1, unit: 'HOURS')
        ansiColor('xterm')
        skipDefaultCheckout()
    }
    agent { label 'master' }
    stages {
        stage('Checkout') {
            steps {
                script {
                    COMMIT = checkoutScmWithRetry(3)
                    echo "COMMIT: ${COMMIT}"
                    VERSION = (sh(returnStdout: true, script: 'git describe --always')).trim()
                    echo "VERSION: '${VERSION}'"
                }
            }
        }

        stage('Analyse') {
            when {
                expression { return isPrecommit }
                beforeOptions true
            }
            steps {
                script {
                    def fileset = changeset
                    def hasCoreM = fileSetMatches(fileset, patternCoreM)
                    def hasCoreA = fileSetMatches(fileset, patternCoreA)
                    def hasCoreValidation = fileSetMatches(fileset, patternCoreValidation)

echo """Change analysis:
- hasCoreM = ${hasCoreM}
- hasCoreA = ${hasCoreA}
- hasCoreValidation = ${hasCoreValidation}
"""

                    if (hasCoreM || hasCoreValidation) {
                        CONFIGURATION['devices'] += CONFIGURATION['mdevices']
                    }
                    if (hasCoreA || hasCoreValidation) {
                        CONFIGURATION['devices'] += CONFIGURATION['adevices']
                    }

                    CORE_VALIDATION &= hasCoreM || hasCoreA || hasCoreValidation
                    
echo """Stage schedule:
- CORE_VALIDATION = ${CORE_VALIDATION}
"""
                }
            }
        }

        stage('CoreValidation') {
            when {
                expression { return CORE_VALIDATION }
                beforeOptions true
            }
            matrix {
                axes {
                    axis {
                      name 'DEVICE'
                      values 'CM0', 'CM0plus', 'CM3', 'CM4', 'CM4FP', 'CM7', 'CM7SP', 'CM7DP',
                             'CM23', 'CM23S', 'CM23NS', 'CM33', 'CM33S', 'CM33NS',
                             'CM35P', 'CM35PS', 'CM35PNS',
                             'CA5', 'CA5neon', 'CA7', 'CA7neon', 'CA9', 'CA9neon'
                    }
                }
                stages {
                    stage('Test') {
                        when {
                            expression { return DEVICE in CONFIGURATION['devices'] }
                            beforeOptions true
                        }
                        agent {
                            kubernetes {
                                defaultContainer 'cmsis'
                                slaveConnectTimeout 600
                                yaml """\
                                    apiVersion: v1
                                    kind: Pod
                                    spec:
                                      imagePullSecrets:
                                        - name: ${dockerinfo_linux['k8sPullSecret']}
                                      securityContext:
                                        runAsUser: 1000
                                        runAsGroup: 1000
                                      containers:
                                        - name: cmsis
                                          image: ${dockerinfo_linux['registryUrl']}/${dockerinfo_linux['image']}:${dockerinfo_linux['label']}
                                          alwaysPullImage: true
                                          imagePullPolicy: Always
                                          command:
                                            - sleep
                                          args:
                                            - infinity
                                          resources:
                                            requests:
                                              cpu: 2
                                              memory: 2Gi
                                    """.stripIndent()
                            }
                        }
                        steps {
                            checkoutScmWithRetry(3)
                            dir('CMSIS/CoreValidation/Tests') {
                                script {
                                    CONFIGURATION['configs'].each { COMPILER, OPTS ->
                                        tee("CV_${COMPILER}_${DEVICE}.log") {
                                            sh "python3 build.py -d ${DEVICE} -c ${COMPILER} -o ${OPTS.join(' -o ')} build run"
                                        }
                                    }
                                }

                                archiveArtifacts artifacts: "CoreValidation_*.zip", allowEmptyArchive: true
                                stash name: "CV_${DEVICE}", includes: '*.log, *.junit'
                            }
                        }
                    }
                }
            }
        }

        stage('Results') {
            steps {
                dir('results') {
                    deleteDir()
                    script {
                        CONFIGURATION['devices'].each { unstash "CV_${it}" }
                    }

                    recordIssues tools: [clang(id: 'AC6', name: 'Arm Compiler 6', pattern: 'CV_AC6_*.log')]
                    recordIssues tools: [clang(id: 'AC6LTM', name: 'Arm Compiler 6 LTM', pattern: 'CV_AC6LTM_*.log')]
                    xunit([
                        JUnit(pattern: 'corevalidation_*.junit', failIfNotNew: false, skipNoTestFiles: true)
                    ])
                }

            }
        }
    }
}
