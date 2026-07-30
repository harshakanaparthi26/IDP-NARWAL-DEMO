// vars/sysdigContainerScan.groovy
//
// Sysdig Secure container image scan  (FIRST DRAFT)
//
// Runs on the python311 agent, which now carries sysdig-cli-scanner and oc.
// Scans the freshly-built image and marks the build UNSTABLE (yellow) on
// policy violations instead of failing it (agreed "warn, don't block" behaviour).
//
// config keys (passed in from the pipeline stage):
//   SERVICE_NAME  - image / app name        (e.g. mlaas-tos-luigi)
//   NAMESPACE     - OpenShift namespace the image was built into
//   IMG_TAG       - tag the build produced   (build.yaml outputs 'latest')
//   TAG           - version, used only in the report file name
//
// optional (sensible defaults):
//   SYSDIG_API_URL        - Sysdig backend         (default https://secure.sysdig.com)
//   SYSDIG_CREDENTIAL_ID  - Jenkins secret-text id  (default 'sysdig')
//   REGISTRY              - registry host to pull the image from

def call(Map config) {

    def apiUrl       = config.SYSDIG_API_URL       ?: 'https://secure.sysdig.com'
    def credentialId = config.SYSDIG_CREDENTIAL_ID ?: 'sysdig'
    // TODO(confirm): the internal OpenShift registry host for your cluster
    def registry     = config.REGISTRY             ?: 'image-registry.openshift-image-registry.svc:5000'
    def imgTag       = config.IMG_TAG              ?: 'latest'

    def imageUrl     = "${registry}/${config.NAMESPACE}/${config.SERVICE_NAME}:${imgTag}"
    def resultsFile  = "${config.SERVICE_NAME}-${config.TAG}-sysdig-scan-result.json"

    // We are already 'oc login'-ed on this agent, so reuse that session token
    // to let the scanner pull the image from the internal registry.
    def registryUser  = sh(returnStdout: true, script: 'oc whoami').trim()
    def registryToken = sh(returnStdout: true, script: 'oc whoami -t').trim()

    withCredentials([string(credentialsId: credentialId, variable: 'SECURE_API_TOKEN')]) {
        echo "Running Sysdig scan on image: ${imageUrl}"

        withEnv(["REGISTRY_USER=${registryUser}", "REGISTRY_PASSWORD=${registryToken}"]) {

            // returnStatus stops a non-zero exit from failing the build outright;
            // we decide what to do with the result below.
            def exitCode = sh(
                returnStatus: true,
                script: """
                    sysdig-cli-scanner \
                        --apiurl ${apiUrl} ${imageUrl} \
                        --output-json=./scan-result.json \
                        --dbpath=/tmp/ \
                        --console-log \
                        --no-cache \
                        --skiptlsverify
                """
            )

            // Always keep the report, whether the scan passed or failed.
            sh "mv scan-result.json ${resultsFile} || true"
            archiveArtifacts artifacts: resultsFile, allowEmptyArchive: true

            // 0 = passed; anything else = policy violations (or a scan error).
            // Warn (yellow) rather than block, per the agreed behaviour.
            if (exitCode != 0) {
                unstable("Sysdig scan flagged issues (exit ${exitCode}) - see report: ${resultsFile}")
            } else {
                echo "Sysdig scan passed - no policy violations."
            }
        }
    }
}
