import subprocess

yadiskLink = "https://yadi.sk/d/11eDCm7Dsn9GA"

# download groundtruth file
command = 'curl ' + '"https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=' \
          + yadiskLink + '&path=/deep1B_groundtruth.ivecs"'
process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
(out, err) = process.communicate()
wgetLink = out.split(',')[0][8:]
wgetCommand = 'wget ' + wgetLink + ' -O deep1B_groundtruth.ivecs'
print "Downloading groundtruth ..."
process = subprocess.Popen(wgetCommand, stdin=subprocess.PIPE, shell=True)
process.stdin.write('e')
process.wait()

# download query file
command = 'curl ' + '"https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=' \
          + yadiskLink + '&path=/deep1B_queries.fvecs"'
process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
(out, err) = process.communicate()
wgetLink = out.split(',')[0][8:]
wgetCommand = 'wget ' + wgetLink + ' -O deep1B_queries.fvecs'
print "Downloading queries ..."
process = subprocess.Popen(wgetCommand, stdin=subprocess.PIPE, shell=True)
process.stdin.write('e')
process.wait()

# download base files
for i in xrange(37):
    command = 'curl ' + '"https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=' \
              + yadiskLink + '&path=/base/base_' + str(i).zfill(2) + '"'
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    (out, err) = process.communicate()
    wgetLink = out.split(',')[0][8:]
    wgetCommand = 'wget ' + wgetLink + ' -O base_' + str(i).zfill(2)
    print "Downloading base chunk " + str(i).zfill(2) + ' ...'
    process = subprocess.Popen(wgetCommand, stdin=subprocess.PIPE, shell=True)
    process.stdin.write('e')
    process.wait()

# download learn files
for i in xrange(14):
    command = 'curl ' + '"https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=' \
              + yadiskLink + '&path=/learn/learn_' + str(i).zfill(2) + '"'
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    (out, err) = process.communicate()
    wgetLink = out.split(',')[0][8:]
    wgetCommand = 'wget ' + wgetLink + ' -O learn_' + str(i).zfill(2)
    print "Downloading learn chunk " + str(i).zfill(2) + ' ...'
    process = subprocess.Popen(wgetCommand, stdin=subprocess.PIPE, shell=True)
    process.stdin.write('e')
    process.wait()